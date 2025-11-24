# Fix: numpy.timedelta64 Time Calculation Error

## Root Cause

All 5,699 trajectories were being rejected with error:
```
'numpy.timedelta64' object has no attribute 'total_seconds'
```

### The Problem

1. **Line 346**: `trajectory_times = trajectory['BaseDateTime'].values`
   - `.values` returns a **numpy array** of datetime64 objects

2. **Line 360**: `time_span = trajectory_times[-1] - trajectory_times[0]`
   - Subtracting two numpy datetime64 objects gives a **numpy.timedelta64** object
   - `numpy.timedelta64` does NOT have a `.total_seconds()` method!

3. **Line 361**: `time_span_hours = time_span.total_seconds() / 3600.0`
   - This fails with `AttributeError` because numpy.timedelta64 doesn't have this method

## The Fix

Convert numpy datetime64 to pandas Timestamp before arithmetic:

```python
# OLD (broken):
time_span = trajectory_times[-1] - trajectory_times[0]  # numpy.timedelta64
time_span_hours = time_span.total_seconds() / 3600.0  # ❌ FAILS

# NEW (fixed):
time_start = pd.Timestamp(trajectory_times[0])  # Convert to pandas
time_end = pd.Timestamp(trajectory_times[-1])   # Convert to pandas
time_span = time_end - time_start                # pandas.Timedelta ✅
time_span_hours = time_span.total_seconds() / 3600.0  # ✅ WORKS
```

Also fixed window comparisons to use pandas Timestamp:
```python
# Convert to pandas for comparisons
trajectory_times_ts = pd.to_datetime(trajectory_times)
window_start = pd.Timestamp(trajectory_times[0])
window_end = window_start + pd.Timedelta(hours=24.0)
window_mask = (trajectory_times_ts >= window_start) & (trajectory_times_ts < window_end)
```

## Why This Matters

With 14 days of data and multiple reports per day:
- Trajectories have **hundreds or thousands of points** (204, 1828, 806, 744, etc.)
- They span **multiple days** (14 days)
- But all were being rejected because the time calculation was failing
- This caused **zero sequences** to be created

## Expected Outcome After Fix

- Trajectories with 2+ points will be accepted
- Time spans will be calculated correctly
- Valid 24-hour windows will be found
- Sequences will be created from valid trajectories

## Files Fixed

- `training/data_loader.py`: Fixed time calculations in validation logic (lines ~358-428)

