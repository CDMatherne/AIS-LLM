# Fix: numpy 'np' Scoping Error

## Root Cause

Error message:
```
'cannot access local variable 'np' where it is not associated with a value'
```

**The Problem:**
1. **Line 604** had: `import numpy as np` (local import inside function)
2. Python's scoping rules: If a variable is assigned anywhere in a function (including imports), Python treats it as **local for the entire function scope**
3. When code tries to use `np.sum()` earlier in the function (lines 381, 406), Python thinks `np` is a local variable that hasn't been assigned yet
4. This causes the error even though `np` is imported at module level (line 7)

**Why This Happens:**
- Python determines variable scope at function definition time (lexical scoping)
- If `import numpy as np` exists anywhere in the function, Python treats `np` as local
- References to `np` before the import fail because the local variable isn't initialized

## The Fix

**Removed:** Local `import numpy as np` on line 604  
**Result:** `np` now comes from module-level import on line 7

## Additional Issue Found

Looking at the log output, there's a deeper problem:

**All 5,699 trajectories are being rejected**, but the trajectories have:
- Many points (1828, 806, 744, 5795, etc.)
- Data spanning 14 days with multiple reports per day

**The Real Issue:**
The validation logic is checking for a 24-hour window with ≥2 points, but the actual requirement is that we need:
1. **Input sequence**: 24 hours of data (sequence_length)
2. **Future data**: 48 hours ahead (prediction_horizon)

For a trajectory spanning 14 days with many points per day:
- It SHOULD have many valid 24-hour windows
- But the validation is rejecting all of them

**Possible causes:**
1. The window checking logic might not be finding valid windows correctly
2. The requirement might be too strict (needs 24h + 48h = 72h total span)
3. The trajectory segmentation might be splitting data in a way that leaves segments too short

## Next Steps

1. ✅ Fixed `np` scoping issue
2. ✅ Fixed numpy timedelta64 calculation
3. ⚠️ Need to verify validation logic is working correctly
4. ⚠️ May need to adjust requirements (24h input + 48h prediction = 72h minimum span)

