# Trajectory Creation Issue - Root Cause and Fix

## Problem
With 14,849 vessels, 0 trajectories/sequences were being created.

## Root Cause
The `TrajectoryProcessor` was initialized with `min_trajectory_hours=24.0`, which meant that during trajectory segmentation, each segment had to independently span at least 24 hours to be considered valid.

When vessels are split by time gaps (max_gap_hours = 6.0), most segments don't meet the 24-hour requirement and are rejected during segmentation, before sequences can even be created.

### Example
A vessel with:
- 10 points over 30 hours total
- A 7-hour gap in the middle

Would create 2 segments:
- Segment 1: 5 points over 12 hours → **REJECTED** (needs 24 hours)
- Segment 2: 5 points over 11 hours → **REJECTED** (needs 24 hours)

Result: 0 trajectories created, even though the vessel has sufficient data.

## Solution
The 24-hour minimum time span check should be applied **later** in the pipeline (during sequence preparation), not during initial trajectory segmentation.

### Changes Made
1. **Trajectory Segmentation Stage**: Use lenient `min_trajectory_hours=1.0` to allow segments to be created
   - This allows segments with at least 3 points and 1 hour span to pass
   - Segments are still split by gaps (>6 hours)

2. **Sequence Preparation Stage**: Apply strict 24-hour check here
   - Trajectories must have >= 3 points AND >= 24 hours span to create sequences
   - This is where the user's requirement is enforced

### Code Changes
- `data_loader.py`: Changed `TrajectoryProcessor` initialization to use `min_trajectory_hours=1.0` during segmentation
- The 24-hour check remains in `prepare_training_sequences()` at line 243

## Expected Result
- More trajectories will be created during segmentation (using 1-hour minimum)
- Sequences will only be created from trajectories that meet the 24-hour requirement
- This allows the pipeline to properly filter data at the appropriate stage

