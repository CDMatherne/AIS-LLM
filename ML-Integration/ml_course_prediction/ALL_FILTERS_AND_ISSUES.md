# All Filters and Values - Complete Analysis

## Critical Bug Found! 🐛

**Line 261 in `data_loader.py`**:
```python
target_end = min(target_start + self.prediction_horizon // 6, len(trajectory))
```

**Problem**: This divides `prediction_horizon` (48 hours) by 6, giving only **8 data points** ahead, not 48 hours!

**Should be**: Looking for data points that are approximately 48 hours in the future, not 8 points ahead.

---

## Complete Filter List

### 1. Data Preprocessing Filters (`data_preprocessing.py`)

| Filter | Parameter | Default Value | Action |
|--------|-----------|---------------|--------|
| **Invalid Timestamps** | - | - | Removes rows where BaseDateTime is NaN |
| **Invalid Positions** | `valid_lat_range` | (-90, 90) | Removes LAT outside range |
| | `valid_lon_range` | (-180, 180) | Removes LON outside range |
| | Zero positions | LAT=0 AND LON=0 | Removes zero positions |
| **Invalid Speeds** | `max_speed_knots` | **100.0 knots** | Removes SOG > 100 |
| | `min_speed_knots` | 0.0 knots | Removes SOG < 0 |
| **Invalid Courses** | `valid_cog_range` | (0, 360) | Removes COG outside range |
| **Unknown Vessel Types** | `filter_unknown_vessel_types` | **False** (preprocessor) | Not filtered in preprocessing |

### 2. Data Loader Filters (`data_loader.py`)

| Filter | Parameter | Default Value | Action |
|--------|-----------|---------------|--------|
| **Unknown Vessel Types** | `filter_unknown_vessel_types` | **True** | Removes VesselType = 0/null |
| **Specific Vessel Types** | `vessel_types` | None (optional) | Filters to specified types if provided |
| **Specific MMSIs** | `mmsi_list` | None (optional) | Filters to specified MMSIs if provided |

### 3. Trajectory Processing Filters (`trajectory_utils.py`)

| Filter | Parameter | Default Value | Action |
|--------|-----------|---------------|--------|
| **Max Gap Hours** | `max_gap_hours` | **6.0 hours** | Splits trajectory if gap > 6h |
| **Min Trajectory Points** | `min_trajectory_points` | **3 points** | Removes trajectories with < 3 points |
| **Min Trajectory Hours** | `min_trajectory_hours` | **1.0 hour** | Removes trajectories with < 1h duration |
| **Outlier Speed** | `max_speed_knots` | **50.0 knots** | Removes positions with speed > 50 kt |
| **Outlier Acceleration** | `max_acceleration` | **10.0 knots/hour** | Removes positions with accel > 10 |

**Note**: TrajectoryProcessor uses **50 knots** max speed, but preprocessing allows **100 knots** - inconsistency!

### 4. Sequence Preparation Filters (`data_loader.py` - `prepare_training_sequences`)

| Filter | Parameter | Default Value | Action | **ISSUE** |
|--------|-----------|---------------|--------|----------|
| **Min Trajectory Length** | `sequence_length + 1` | **25 points** | Skips if trajectory < 25 points | ⚠️ May be too strict |
| **Outlier Filtering** | Applied before sequence prep | See above | Removes outliers first | ⚠️ May remove too much |
| **Target End Calculation** | `prediction_horizon // 6` | **8 points** | Only looks 8 points ahead | 🐛 **BUG: Should be 48 hours!** |
| **Time Difference Check** | `abs(time_diff - 48) > 12` | ±12 hours | Skips if not 36-60h ahead | ⚠️ Very strict |
| **Sliding Window Step** | Hardcoded | **6 hours** | Creates sequences every 6h | ⚠️ May skip valid sequences |
| **Feature Extraction** | - | - | Skips if feature extraction fails | - |

---

## Critical Issues Identified

### 🐛 Bug #1: Target End Calculation (CRITICAL)
**Location**: `data_loader.py` line 261

**Current Code**:
```python
target_end = min(target_start + self.prediction_horizon // 6, len(trajectory))
# prediction_horizon = 48
# 48 // 6 = 8
# So it only looks 8 data points ahead, not 48 hours!
```

**Problem**: 
- If data has 1 point per hour, this only looks 8 hours ahead (not 48!)
- If data has 1 point per 6 hours, this looks 48 hours ahead (accidentally correct)
- If data has multiple points per hour, this looks much less than 48 hours

**Fix Needed**: Calculate target based on **time**, not **number of points**

### ⚠️ Issue #2: Time Difference Tolerance Too Strict
**Location**: `data_loader.py` line 273

**Current Code**:
```python
if abs(time_diff - self.prediction_horizon) > 12:  # Allow 12h tolerance
    continue
```

**Problem**: 
- Requires target to be exactly 36-60 hours ahead
- If trajectory doesn't have data at exactly 48 hours, sequence is skipped
- Very strict - may filter out many valid sequences

**Fix Needed**: Increase tolerance or make it configurable

### ⚠️ Issue #3: Sequence Length Requirement
**Location**: `data_loader.py` line 243

**Current Code**:
```python
if len(trajectory) < self.sequence_length + 1:
    continue
# sequence_length = 24, so requires 25 points
```

**Problem**: 
- Requires 25 data points minimum
- If data is sparse (e.g., 1 point per 2 hours), need 50 hours of data
- May filter out many valid trajectories

**Fix Needed**: Check both point count AND time span

### ⚠️ Issue #4: Speed Filter Inconsistency
**Location**: Multiple files

**Problem**: 
- Preprocessing allows up to **100 knots**
- TrajectoryProcessor filters out speeds > **50 knots**
- Fast vessels (HSC, passenger) may be incorrectly filtered

**Fix Needed**: Make consistent or vessel-type aware

### ⚠️ Issue #5: Sliding Window Step Too Large
**Location**: `data_loader.py` line 253

**Current Code**:
```python
for start_idx in range(0, max_start, 6):  # Slide by 6 hours
```

**Problem**: 
- Creates sequences every 6 hours
- May miss valid sequences between windows
- Reduces number of training examples

**Fix Needed**: Make configurable, default to smaller step (e.g., 1 hour)

---

## Recommended Fixes (Priority Order)

### 1. Fix Target End Calculation (CRITICAL - Bug)
```python
# CURRENT (WRONG):
target_end = min(target_start + self.prediction_horizon // 6, len(trajectory))

# FIXED:
# Find target_end such that time difference is approximately prediction_horizon
target_time = input_seq['BaseDateTime'].iloc[-1] + pd.Timedelta(hours=self.prediction_horizon)
# Find index where time is closest to target_time
time_diffs = (trajectory['BaseDateTime'] - target_time).abs()
target_end = time_diffs.idxmin() + 1  # +1 to include the point
target_end = min(target_end, len(trajectory))
```

### 2. Relax Time Difference Tolerance
```python
# CURRENT:
if abs(time_diff - self.prediction_horizon) > 12:  # ±12 hours

# FIXED:
tolerance = self.prediction_horizon * 0.5  # 50% tolerance (24 hours for 48h horizon)
if abs(time_diff - self.prediction_horizon) > tolerance:
    continue
```

### 3. Fix Sequence Length Check
```python
# CURRENT:
if len(trajectory) < self.sequence_length + 1:  # 25 points

# FIXED: Check both points and time span
time_span_hours = (trajectory['BaseDateTime'].max() - trajectory['BaseDateTime'].min()).total_seconds() / 3600
if len(trajectory) < 10 or time_span_hours < self.sequence_length * 0.5:
    continue
```

### 4. Make Sliding Window Configurable
```python
# Add parameter:
sliding_window_step_hours: int = 1  # Default 1 hour instead of 6

# Use:
for start_idx in range(0, max_start, sliding_window_step_hours):
```

### 5. Fix Speed Filter Inconsistency
```python
# In TrajectoryProcessor, use same max_speed as preprocessing
# Or make it vessel-type aware
max_speed = 100.0  # Match preprocessing default
# Or: max_speed = get_max_speed_for_vessel_type(vessel_type)
```

---

## Current Filter Values Summary Table

| Stage | Filter | Value | Impact |
|-------|--------|-------|--------|
| **Preprocessing** | Max Speed | 100.0 kt | Removes ~0.2% |
| **Preprocessing** | Invalid Positions | LAT/LON ranges | Removes invalid |
| **Preprocessing** | Invalid Courses | 0-360° | Removes invalid |
| **Data Loader** | Unknown Vessel Types | Filter=True | Removes ~0.5% |
| **Trajectory** | Max Gap | 6.0 hours | Splits trajectories |
| **Trajectory** | Min Points | 3 points | Removes short |
| **Trajectory** | Min Hours | 1.0 hour | Removes short |
| **Trajectory** | Max Speed | **50.0 kt** | ⚠️ Inconsistent with preprocessing |
| **Trajectory** | Max Acceleration | 10.0 kt/h | Removes outliers |
| **Sequence** | Min Length | **25 points** | ⚠️ May be too strict |
| **Sequence** | Target Calculation | **8 points** | 🐛 **BUG: Should be 48 hours!** |
| **Sequence** | Time Tolerance | ±12 hours | ⚠️ Very strict |
| **Sequence** | Sliding Step | **6 hours** | ⚠️ May skip sequences |

---

## Immediate Action Required

1. **Fix Bug #1** (Target end calculation) - This is likely why no sequences are created
2. **Relax time tolerance** - Currently too strict
3. **Fix speed inconsistency** - TrajectoryProcessor too restrictive
4. **Add diagnostic logging** - To see exactly where sequences are filtered

---

**Status**: Critical bug identified in sequence preparation  
**Priority**: Fix target end calculation immediately

