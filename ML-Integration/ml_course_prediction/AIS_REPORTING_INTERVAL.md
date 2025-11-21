# AIS Reporting Interval - Project Configuration

## Key Information

**AIS Reporting Interval**: Every **3 hours**

**Implications for 24-Hour Windows**:
- **Maximum points per 24-hour window**: 8 points (24 hours ÷ 3 hours = 8)
- **Minimum points for valid sequence**: 2 points (6 hours of data)
- **Typical sequence length**: 2-8 points per 24-hour window

## Updated Project Parameters

### Trajectory Processing
- **`min_trajectory_points`**: Changed from 3 to **2 points**
  - Reason: With 3-hour reporting, 2 points = 6 hours of data (minimum reasonable)
  - Location: `utils/trajectory_utils.py`

### Sequence Preparation
- **`min_points_in_24h`**: Changed from 3 to **2 points**
  - Reason: Minimum 2 points (6 hours) for valid 24-hour window
  - Location: `training/data_loader.py`

### Feature Engineering
- **Sequence length**: 24 hours
- **Expected points per sequence**: 2-8 points
- **No fixed point count requirement**: Sequences use all available points in the 24-hour window
- Location: `utils/feature_engineering.py`

## Model Architecture Considerations

When implementing the model, account for:
1. **Variable-length sequences**: 2-8 points per input sequence
2. **Padding strategy**: May need to pad shorter sequences to fixed length, or use variable-length RNN/Transformer
3. **Feature extraction**: Must work with sparse data (not every hour has a report)

## Data Characteristics

- **Reporting frequency**: Every 3 hours
- **24-hour window**: Contains 2-8 AIS reports
- **48-hour prediction**: Will have 2-8 target points (depending on future data availability)
- **Gap tolerance**: 6-hour gaps are normal (2 reporting intervals)

## Code Constants

```python
AIS_REPORTING_INTERVAL_HOURS = 3.0
MAX_POINTS_PER_24H = 8  # 24 / 3 = 8
MIN_POINTS_PER_24H = 2  # Minimum for valid sequence (6 hours)
```

## Impact on Training

- **Input sequences**: Variable length (2-8 points)
- **Batch processing**: May need padding or variable-length batching
- **Model input**: Should handle sequences of different lengths
- **Feature engineering**: Must extract meaningful features from sparse temporal data

