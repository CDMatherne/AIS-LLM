# Training Pipeline Analysis - Mismatch Detection

## Issues Found and Fixed

### 1. ✅ FIXED: Positional Encoding Sequence Length Mismatch
**Problem:**
- Positional encoding had hardcoded `max_seq_len=10`
- Actual sequences can be 1304+ time steps long
- Caused `RuntimeError: The size of tensor a (1304) must match the size of tensor b (10)`

**Fix:**
- Modified `PositionalEncoding.forward()` to dynamically generate positional encodings for sequences longer than pre-computed buffer
- Increased default `max_seq_len` from 10 to 2000 in transformer encoder
- Positional encoding now handles variable-length sequences correctly

**Files Modified:**
- `models/architectures/transformer_encoder.py`
- `models/architectures/hybrid_model.py`

---

## Potential Issues Identified

### 2. ⚠️ Sequence Length Variability
**Current State:**
- Sequences can have extremely variable lengths (2 to 1304+ points)
- 24-hour window with AIS reporting every 3 hours should = max 8 points
- But actual sequences are much longer, suggesting:
  - Either interpolation is happening somewhere (not found)
  - Or trajectories have higher-frequency data
  - Or sequences span longer than 24 hours somehow

**Recommendation:**
- Add sequence length validation/logging
- Consider clipping sequences to a maximum length if they exceed expected bounds
- Check why sequences are so long (may indicate data issue)

**Location:**
- `training/data_loader.py` - `prepare_training_sequences()`
- Sequences created from trajectories filtered to 24-hour windows

### 3. ⚠️ Feature Dimension Consistency
**Current State:**
- Features extracted: LAT, LON, SOG, COG, Heading = 5 features
- Model `input_size` determined dynamically from first sequence
- Should be consistent but verify in practice

**Verification:**
- ✅ Feature engineering always extracts same columns
- ✅ Model gets `input_size` from actual sequences
- ⚠️ No validation that all sequences have same feature count

**Recommendation:**
- Add validation in `collate_sequences()` to ensure consistent feature dimensions
- Log feature dimensions for first few sequences

**Location:**
- `utils/feature_engineering.py` - `create_sequence_features()`
- `training/trainer.py` - `collate_sequences()`

### 4. ✅ VERIFIED: Target Dimensions
**Current State:**
- Position targets: (B, 2) [LAT, LON] ✅
- Speed targets: (B, 1) [SOG] ✅
- Course targets: (B, 1) [COG] ✅
- Model outputs match these shapes ✅

**Status:** No issues found

### 5. ✅ VERIFIED: LSTM Variable-Length Handling
**Current State:**
- Uses `pack_padded_sequence` correctly ✅
- Handles sorting/unsorting properly ✅
- Lengths tensor used correctly ✅

**Status:** No issues found (previous bug was fixed)

### 6. ✅ VERIFIED: Transformer Mask Creation
**Current State:**
- Mask created as: `torch.arange(max_len) < lengths.unsqueeze(1)` ✅
- Mask inverted correctly: `src_key_padding_mask = ~mask.bool()` ✅
- Transformer receives correct padding mask ✅

**Status:** No issues found

### 7. ✅ VERIFIED: Loss Function Input Shapes
**Current State:**
- Position predictions: Dict with 'mean' (B, 2), 'std' (B, 2) ✅
- Targets: (B, 2) ✅
- Loss computation handles shapes correctly ✅
- Physics loss handles None input_sequences gracefully ✅

**Status:** No issues found

### 8. ⚠️ Batch Padding and Memory
**Current State:**
- Batches padded to max sequence length in batch
- If one sequence is 1304 points, entire batch is padded to 1304
- Can cause memory issues with large batches

**Recommendation:**
- Consider max sequence length clipping
- Or batch sequences by similar lengths
- Or use smaller batches for long sequences

**Location:**
- `training/trainer.py` - `collate_sequences()`

---

## Sequence Length Investigation Needed

### Root Cause Analysis Required:
1. **Why are sequences so long?**
   - Expected: 2-8 points per 24-hour window (AIS reports every 3h)
   - Actual: Up to 1304 points
   - Possible causes:
     - Data has higher frequency than expected
     - Interpolation happening somewhere
     - Time window filtering not working correctly
     - Multiple trajectory segments being combined

2. **Check Sequence Generation:**
   ```python
   # In data_loader.py, line ~412
   input_seq = trajectory[
       (trajectory['BaseDateTime'] >= current_start_time) &
       (trajectory['BaseDateTime'] < input_end_time)
   ].copy()
   ```
   - This should filter to 24-hour window
   - But if trajectory has 1-minute intervals, that's 1440 points!

3. **Recommendation:**
   - Add logging to show actual sequence length distribution
   - Add max sequence length limit in sequence creation
   - Consider downsampling/interpolation if sequences are too long

---

## Summary of Fixes Applied

1. ✅ **Positional Encoding**: Made dynamic to handle variable-length sequences
2. ✅ **Max Seq Len**: Increased from 10 to 2000 (dynamic encoding handles longer)

## Recommendations for Future

1. **Add Sequence Length Validation:**
   - Log sequence length statistics
   - Add max length limit (e.g., 500 points)
   - Investigate why sequences are so long

2. **Add Feature Dimension Validation:**
   - Verify all sequences have same feature count
   - Log feature dimensions

3. **Batch Size Optimization:**
   - Consider grouping sequences by length
   - Use smaller batches for very long sequences

4. **Add Debug Logging:**
   - Log actual sequence lengths during training
   - Log feature dimensions
   - Log batch statistics

---

## Files to Monitor

- `training/data_loader.py` - Sequence creation
- `training/trainer.py` - Batching and training
- `models/architectures/transformer_encoder.py` - Positional encoding
- `models/architectures/lstm_encoder.py` - Variable-length handling
- `models/architectures/hybrid_model.py` - Overall model integration

