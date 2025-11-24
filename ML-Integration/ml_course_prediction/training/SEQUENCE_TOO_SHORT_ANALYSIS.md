# Analysis: "Sequences Too Short" Error

## Problem Summary

The training run is skipping all trajectories (2928 trajectories) with the error "too short", resulting in 0 training sequences being created.

**Log Evidence:**
```
Skipped 2928 trajectories: too short
WARNING: No training sequences created! Check filtering parameters.
Sequence length required: 24 hours
Prediction horizon: 48 hours
```

## Root Cause Analysis

### Issue Location
File: `training/data_loader.py`
Lines: 340-381

### The Problem

The validation logic is **too strict** and incorrectly filters out valid trajectories. Here's the flawed logic:

1. **Line 343**: Calculates `time_span_hours` using:
   ```python
   time_span_hours = (trajectory_times[-1] - trajectory_times[0]) / pd.Timedelta(hours=1)
   ```
   
   **BUG**: This division operation may not work correctly with pandas Timestamps. It should use `.total_seconds() / 3600` instead.

2. **Line 346**: Requires trajectories to span **at least 24 hours TOTAL**:
   ```python
   if time_span_hours >= 24.0 and len(trajectory) >= min_points_in_24h:
   ```
   
   **PROBLEM**: This filters out valid trajectories that:
   - Have enough points (≥2) within a 24-hour window
   - But don't span a full 24 hours from first to last point
   - Example: A trajectory with 3 points over 6 hours should still be valid

3. **Lines 356-371**: Fallback check only runs if `time_span_hours < 48`:
   ```python
   if not has_valid_24h_window and time_span_hours < 48:
   ```
   
   **PROBLEM**: If the time span calculation is wrong (from bug #1), this fallback never executes.

### What Should Happen

The code should check if **ANY** 24-hour sliding window within the trajectory has at least 2 points. It should NOT require the entire trajectory to span 24 hours.

## Example of Problematic Behavior

**Valid Trajectory (but incorrectly filtered):**
- Trajectory has 5 points
- Points span 6 hours total (not 24 hours)
- All points are within a 6-hour window
- **Current code**: Filters this out (doesn't span 24 hours)
- **Should be**: Keep this (has 5 points in a 6-hour window, which is a valid sequence)

## Fix Required

1. **Fix time span calculation** to use `.total_seconds() / 3600`
2. **Remove requirement** that trajectory must span 24 hours total
3. **Improve window checking** to properly find any valid 24-hour window
4. **Add better logging** to show why trajectories are being filtered

