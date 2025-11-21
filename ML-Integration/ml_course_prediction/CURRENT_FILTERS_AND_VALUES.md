# Current Filters and Values - Complete List

## Overview
This document lists ALL filters currently applied in the ML course prediction data pipeline and their default values.

---

## 1. Data Preprocessing Filters (`data_preprocessing.py`)

### Position Filters
- **Filter**: Invalid latitude/longitude
- **Values**:
  - `valid_lat_range`: (-90, 90) degrees
  - `valid_lon_range`: (-180, 180) degrees
  - `not_zero`: Filters positions where LAT=0 AND LON=0
- **Action**: Removes records with invalid positions
- **Default**: Active

### Speed Filters
- **Filter**: Invalid speed over ground (SOG)
- **Values**:
  - `max_speed_knots`: 100.0 knots (default)
  - `min_speed_knots`: 0.0 knots (default)
- **Action**: Removes records where SOG < 0 or SOG > 100
- **Note**: NaN values are kept (valid for stationary vessels)
- **Default**: Active

### Course Filters
- **Filter**: Invalid course over ground (COG)
- **Values**:
  - `valid_cog_range`: (0, 360) degrees
- **Action**: Removes records where COG < 0 or COG > 360
- **Note**: NaN values are kept (valid for stationary vessels)
- **Default**: Active

### Timestamp Filters
- **Filter**: Invalid timestamps
- **Action**: Removes rows where BaseDateTime is NaN or invalid
- **Default**: Active

### Vessel Type Filters (Preprocessing)
- **Filter**: Unknown vessel types
- **Values**:
  - `filter_unknown_vessel_types`: False (default in preprocessor)
- **Action**: Filters out VesselType = 0, 0.0, or NaN
- **Default**: **INACTIVE** (filtering happens in data loader instead)

---

## 2. Data Loader Filters (`data_loader.py`)

### Unknown Vessel Type Filter
- **Filter**: Unknown vessel types (VesselType = 0, 0.0, or null)
- **Values**:
  - `filter_unknown_vessel_types`: **True** (default)
- **Action**: Removes all records with unknown vessel types
- **Impact**: Removes ~0.5% of data (39K records from 7.3M)
- **Default**: **ACTIVE**

### Vessel Type Filter (Specific Types)
- **Filter**: Specific vessel type codes
- **Values**:
  - `vessel_types`: Optional list (e.g., [70] for cargo)
- **Action**: Filters to only specified vessel types
- **Default**: **INACTIVE** (no filtering unless specified)

### MMSI Filter
- **Filter**: Specific vessel MMSIs
- **Values**:
  - `mmsi_list`: Optional list of MMSI values
- **Action**: Filters to only specified vessels
- **Default**: **INACTIVE** (no filtering unless specified)

---

## 3. Trajectory Processing Filters (`trajectory_utils.py`)

### Trajectory Segmentation
- **Filter**: Trajectories split by time gaps
- **Values**:
  - `max_gap_hours`: **6.0 hours** (default)
  - `min_trajectory_points`: **3 points** (default)
  - `min_trajectory_hours`: **1.0 hour** (default)
- **Action**: 
  - Splits trajectory if time gap > 6 hours
  - Removes trajectories with < 3 points
  - Removes trajectories with < 1 hour duration
- **Default**: Active

### Outlier Filtering
- **Filter**: Physically implausible positions
- **Values**:
  - `max_speed_knots`: **50.0 knots** (default in TrajectoryProcessor)
  - `max_acceleration`: **10.0 knots/hour** (default)
- **Action**: Removes positions with:
  - Speed > 50 knots (calculated from position changes)
  - Acceleration > 10 knots/hour
- **Note**: This is DIFFERENT from the preprocessing max_speed (100 knots)
- **Default**: Active

---

## 4. Sequence Preparation Filters (`data_loader.py` - `prepare_training_sequences`)

### Minimum Trajectory Length
- **Filter**: Trajectories too short for sequences
- **Values**:
  - `sequence_length`: **24 hours** (default)
  - Required: `len(trajectory) >= sequence_length + 1`
- **Action**: Skips trajectories with < 25 data points
- **Default**: Active

### Sliding Window Step
- **Filter**: Window step size
- **Values**:
  - Step size: **6 hours** (hardcoded)
- **Action**: Creates sequences every 6 hours
- **Impact**: Reduces number of sequences (but doesn't filter them out)

### Target Sequence Availability
- **Filter**: Insufficient future data
- **Values**:
  - `prediction_horizon`: **48 hours** (default)
  - Required: Must have data at least `prediction_horizon` hours ahead
- **Action**: Skips sequences where target_end <= target_start
- **Default**: Active

### Time Difference Tolerance
- **Filter**: Target time difference
- **Values**:
  - `prediction_horizon`: **48 hours**
  - Tolerance: **12 hours** (hardcoded)
- **Action**: Skips sequences where `abs(time_diff - 48) > 12`
- **Impact**: Only keeps sequences where target is 36-60 hours ahead
- **Default**: Active

### Feature Extraction Failure
- **Filter**: Failed feature extraction
- **Action**: Skips sequences where `create_sequence_features()` returns empty
- **Default**: Active

---

## 5. Summary of All Active Filters

### Preprocessing Stage
1. ✅ Invalid timestamps → Removed
2. ✅ Invalid positions (LAT/LON out of range or 0,0) → Removed
3. ✅ Invalid speeds (SOG < 0 or > 100 knots) → Removed
4. ✅ Invalid courses (COG < 0 or > 360) → Removed
5. ❌ Unknown vessel types → **NOT filtered in preprocessing** (filtered in data loader)

### Data Loader Stage
6. ✅ Unknown vessel types (VesselType = 0/null) → Removed (default: True)
7. ⚠️ Specific vessel types → Filtered if `vessel_types` parameter provided
8. ⚠️ Specific MMSIs → Filtered if `mmsi_list` parameter provided

### Trajectory Processing Stage
9. ✅ Trajectories with gaps > 6 hours → Split into separate trajectories
10. ✅ Trajectories with < 3 points → Removed
11. ✅ Trajectories with < 1 hour duration → Removed
12. ✅ Outlier positions (speed > 50 knots or acceleration > 10) → Removed

### Sequence Preparation Stage
13. ✅ Trajectories with < 25 points → Skipped
14. ✅ Sequences without 48-hour future data → Skipped
15. ✅ Sequences where target time is not 36-60 hours ahead → Skipped
16. ✅ Sequences where feature extraction fails → Skipped

---

## Critical Filter Values

### Most Restrictive Filters (Likely Causes of No Sequences)

1. **Sequence Length Requirement**:
   - Requires: `len(trajectory) >= 25 points`
   - **Default**: 24 hours of history
   - **Issue**: If data is sparse (few points per day), trajectories may be too short

2. **48-Hour Prediction Horizon**:
   - Requires: Data available 48 hours after input sequence end
   - **Tolerance**: ±12 hours (36-60 hours acceptable)
   - **Issue**: If trajectories are short or have gaps, may not have 48h of future data

3. **Time Difference Tolerance**:
   - Requires: Target must be 36-60 hours ahead
   - **Hardcoded**: 12-hour tolerance
   - **Issue**: Very strict - may filter out valid sequences

4. **Outlier Filtering**:
   - Removes: Speed > 50 knots or acceleration > 10
   - **Issue**: May be too aggressive, especially for fast vessels (HSC, passenger)

5. **Trajectory Segmentation**:
   - Splits: If gap > 6 hours
   - **Issue**: Creates many short trajectories that may not meet sequence length requirement

---

## Recommended Fixes

### 1. Relax Sequence Requirements
```python
# Current: Requires 25+ points
if len(trajectory) < self.sequence_length + 1:
    continue

# Suggested: Reduce requirement or make configurable
if len(trajectory) < max(10, self.sequence_length // 2):
    continue
```

### 2. Relax Time Difference Tolerance
```python
# Current: ±12 hours tolerance (36-60 hours)
if abs(time_diff - self.prediction_horizon) > 12:
    continue

# Suggested: Increase tolerance or make configurable
tolerance = self.prediction_horizon * 0.5  # 50% tolerance
if abs(time_diff - self.prediction_horizon) > tolerance:
    continue
```

### 3. Adjust Outlier Filtering
```python
# Current: max_speed_knots = 50.0 in TrajectoryProcessor
# But preprocessing allows up to 100.0

# Suggested: Make consistent or vessel-type aware
max_speed_knots = 100.0  # Match preprocessing
# Or: vessel_type_aware_max_speed
```

### 4. Reduce Sliding Window Step
```python
# Current: Step by 6 hours
for start_idx in range(0, max_start, 6):

# Suggested: Step by 1 hour for more sequences
for start_idx in range(0, max_start, 1):
```

### 5. Make Filters Configurable
Add configuration options for:
- Minimum trajectory length
- Time difference tolerance
- Outlier filtering thresholds
- Sliding window step size

---

## Current Default Values Summary

| Filter | Parameter | Default Value | Location |
|--------|-----------|---------------|----------|
| Max Speed (Preprocessing) | `max_speed_knots` | 100.0 knots | `data_preprocessing.py` |
| Max Speed (Trajectory) | `max_speed_knots` | 50.0 knots | `trajectory_utils.py` |
| Max Acceleration | `max_acceleration` | 10.0 knots/hour | `trajectory_utils.py` |
| Max Gap Hours | `max_gap_hours` | 6.0 hours | `trajectory_utils.py`, `data_loader.py` |
| Min Trajectory Points | `min_trajectory_points` | 3 points | `trajectory_utils.py` |
| Min Trajectory Hours | `min_trajectory_hours` | 1.0 hour | `trajectory_utils.py` |
| Sequence Length | `sequence_length` | 24 hours | `data_loader.py` |
| Prediction Horizon | `prediction_horizon` | 48 hours | `data_loader.py` |
| Time Tolerance | Hardcoded | ±12 hours | `data_loader.py` |
| Sliding Step | Hardcoded | 6 hours | `data_loader.py` |
| Filter Unknown Types | `filter_unknown_vessel_types` | True | `data_loader.py` |

---

## Next Steps

1. **Review sequence preparation logic** - Most likely issue
2. **Relax time difference tolerance** - Currently very strict
3. **Reduce minimum trajectory length** - May be too restrictive
4. **Adjust outlier filtering** - May be removing valid fast vessels
5. **Add diagnostic logging** - To see exactly where sequences are being filtered

---

**Status**: All filters documented  
**Issue**: Sequences being filtered out - likely due to strict sequence requirements

