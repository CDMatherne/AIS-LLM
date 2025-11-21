# Unknown Vessel Types Handling

## Current Behavior

**Unknown vessel types (VesselType = 0) are NOT filtered by default** in the preprocessing step, but **ARE filtered by default** in the data loader.

### Preprocessing (`data_preprocessing.py`)
- **Default**: Unknown vessel types are **kept** (not filtered)
- Missing VesselType values are filled with `0` (unknown)
- Option: `filter_unknown_vessel_types=False` (default)

### Data Loader (`data_loader.py`)
- **Default**: Unknown vessel types are **filtered out**
- Option: `filter_unknown_vessel_types=True` (default)

## Why Filter Unknown Vessel Types?

### Reasons to Filter:
1. **Model Training**: Unknown types don't provide useful information
2. **Pattern Consistency**: Unknown types may have inconsistent patterns
3. **Data Quality**: Vessels with missing type info may have other data quality issues
4. **Prediction Accuracy**: Model can't learn type-specific patterns for unknown types

### Reasons to Keep:
1. **Data Completeness**: Don't lose potentially useful data
2. **Anomaly Detection**: Unknown types might be interesting for anomaly detection
3. **Research**: May want to analyze unknown types separately

## Configuration

### Data Loader (Recommended)
```python
# Filter unknown types (default)
df = await loader.load_training_data(
    start_date="2024-01-01",
    end_date="2024-01-07",
    filter_unknown_vessel_types=True  # Default: True
)

# Keep unknown types
df = await loader.load_training_data(
    start_date="2024-01-01",
    end_date="2024-01-07",
    filter_unknown_vessel_types=False
)
```

### Preprocessor (Optional)
```python
# Filter unknown types during preprocessing
preprocessor = HistoricalDataPreprocessor(
    data_path=r"C:\AIS_Data_Testing\Historical\2024",
    filter_unknown_vessel_types=True  # Default: False
)
```

## Statistics

To see how many unknown vessel types are in your data:

```python
df = await loader.load_training_data(
    start_date="2024-01-01",
    end_date="2024-01-01",
    filter_unknown_vessel_types=False  # Keep them to count
)

unknown_count = (df['VesselType'] == 0).sum()
unknown_pct = (unknown_count / len(df) * 100) if len(df) > 0 else 0

print(f"Unknown vessel types: {unknown_count:,} ({unknown_pct:.2f}%)")
```

## Recommendation

**For ML Model Training**: **Filter unknown vessel types** (default behavior in data loader)
- Ensures clean training data
- Model learns from known vessel types only
- Better prediction accuracy

**For Data Analysis**: Keep unknown types to analyze separately
- May reveal data quality issues
- Could identify misclassified vessels
- Useful for data exploration

## Implementation Details

### Data Loader
- Filters unknown types **after** preprocessing
- Logs how many records were filtered
- Default: `filter_unknown_vessel_types=True`

### Preprocessor
- Can filter during preprocessing (optional)
- Default: `filter_unknown_vessel_types=False`
- Fills missing values with 0 first, then optionally filters

## Notes

- Unknown vessel types are assigned `VesselType = 0`
- Filtering happens after all other preprocessing steps
- Filtering is logged for transparency
- Can be disabled if needed for analysis

---

**Status**: Unknown vessel types are filtered by default in data loader  
**Recommendation**: Keep default (filter=True) for model training

