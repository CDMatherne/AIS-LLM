# Root Cause Analysis: No Sequences Created

## Summary

All 5,699 trajectories were rejected:
- **2,735 trajectories**: Skipped as "too short" (validation failed)
- **2,964 sequences**: Skipped due to "no future data" (48h prediction horizon not available)

**Total result:** 0 sequences created

## Root Cause #1: numpy 'np' Scoping Error (FIXED)

**Error:** `'cannot access local variable 'np' where it is not associated with a value'`

**Problem:** Local `import numpy as np` on line 604 caused Python to treat `np` as local for entire function
**Fix:** Removed redundant local import (numpy already imported at module level, line 7)

## Root Cause #2: Validation Logic Issue

### The Problem

**Validation checks for:**
1. At least 2 points within a 24-hour window
2. But doesn't check if trajectory can actually create sequences with:
   - 24 hours of input data (sequence_length)
   - 48 hours of future data for prediction (prediction_horizon)

**Total requirement:** trajectory needs to span at least **72 hours** (24h input + 48h prediction) to create sequences!

### Current Flow

1. **Trajectory Segmentation** (trajectory_utils.py):
   - Splits trajectories when gaps > 6 hours
   - Trajectories spanning 14 days might be split into many segments
   - Each segment might be shorter than 72 hours

2. **Validation** (data_loader.py, lines 343-446):
   - Checks if trajectory has ≥2 points in a 24h window
   - But doesn't check if trajectory spans enough time for sequence creation
   - **Problem:** A trajectory spanning 30 hours will pass validation (has 24h window), but can't create sequences (needs 24h input + 48h future = 72h total)

3. **Sequence Creation** (data_loader.py, lines 454-580):
   - Creates input sequences of 24 hours
   - Needs future data 48 hours ahead
   - **Problem:** If trajectory only spans 30-60 hours, it can't provide 48h future data

### Why Trajectories Are Being Rejected

**Example trajectory spanning 14 days:**
- After segmentation, might be split into segments of 3-5 days each
- Each segment has many points (1828, 806, 744, etc.)
- Each segment passes validation (has 24h windows)
- But when creating sequences:
  - Last 48 hours of trajectory can't have future data
  - Sequences that end within 48h of trajectory end get skipped
  - This might be why many sequences are skipped

### The Real Issue

The validation checks if a trajectory **could theoretically** have valid windows, but doesn't check if it **actually can** create sequences given the prediction horizon requirement.

## Fixes Needed

1. ✅ **Fixed:** numpy scoping error (removed local import)
2. ✅ **Fixed:** numpy timedelta64 calculation (convert to pandas Timestamp)
3. ⚠️ **Need to fix:** Validation should check for minimum trajectory span (sequence_length + prediction_horizon = 72 hours)
4. ⚠️ **Need to fix:** Handle trajectories that are long enough but sequences near the end get skipped

## Recommended Changes

1. **Update validation logic** to check if trajectory spans at least `sequence_length + prediction_horizon` hours
2. **Or** relax the prediction horizon requirement for sequences near the end of trajectories
3. **Or** adjust trajectory segmentation to keep longer segments together

