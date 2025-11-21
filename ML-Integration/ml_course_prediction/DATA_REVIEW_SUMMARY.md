# Historical Data Review and Preprocessing Summary

## Data Source Review

**Location**: `C:\AIS_Data_Testing\Historical\2024`  
**Files**: 366 parquet files (one per day of 2024)  
**Volume**: ~7.3 million records per day  
**Total**: ~2.67 billion records for full year

### Data Structure
- **Format**: Parquet files named `AIS_YYYY_MM_DD.parquet`
- **Columns**: 17 fields including all required AIS data
- **Quality**: High quality with minimal missing data in required fields

## Identified Transformations Required

### ✅ Implemented Transformations

1. **DateTime Conversion**
   - BaseDateTime stored as string → Convert to pandas datetime
   - Handle invalid date formats gracefully

2. **Position Validation**
   - Filter invalid LAT (-90 to 90)
   - Filter invalid LON (-180 to 180)
   - Remove zero positions (0, 0)

3. **Speed Validation**
   - Filter SOG > 100 knots (~0.2% of records)
   - Allow NaN for stationary vessels
   - Configurable speed limits

4. **Course Validation**
   - Filter COG outside 0-360 degrees
   - Allow NaN for stationary vessels

5. **Missing Value Handling**
   - VesselType: Fill with 0 (unknown)
   - Length/Width/Draft: Fill with 0
   - Preserve NaN where valid (SOG, COG, Heading)

6. **Data Sorting**
   - Sort by MMSI and BaseDateTime
   - Required for trajectory processing

## Components Created

### 1. HistoricalDataPreprocessor (`utils/data_preprocessing.py`)
- Loads parquet files from historical directory
- Handles date ranges and single files
- Comprehensive data validation and cleaning
- Configurable filtering parameters
- Data quality summary generation

**Key Methods**:
- `load_date_file()`: Load single day
- `load_date_range()`: Load multiple days
- `preprocess_dataframe()`: Clean and validate data
- `get_data_summary()`: Generate statistics

### 2. Updated CoursePredictionDataLoader
- Integrated historical data preprocessor
- Priority system: Historical data → Cache → Connector
- Automatic preprocessing option
- Enhanced filtering and logging

### 3. Test Script (`training/test_data_loading.py`)
- Comprehensive testing of data loading
- Tests single day, date range, filtering
- Validates preprocessing pipeline
- Tests sequence preparation

### 4. Documentation
- `DATA_TRANSFORMATIONS.md`: Detailed transformation guide
- `DATA_REVIEW_SUMMARY.md`: This summary
- Updated configuration with data path

## Data Quality After Preprocessing

### Expected Results
- **Valid positions**: 100% within valid ranges
- **Valid speeds**: 100% within 0-100 knots (or NaN)
- **Valid courses**: 100% within 0-360 degrees (or NaN)
- **Data loss**: ~0.2-0.5% (invalid records removed)
- **Sorted**: By MMSI and time
- **Typed**: Correct data types

### Sample Statistics (2024-01-01)
- **Raw records**: 7,296,275
- **Unique vessels**: 14,868
- **Invalid speeds**: 16,633 (0.23%)
- **All other fields**: Valid

## Usage

### Basic Usage
```python
from ml_course_prediction.training.data_loader import CoursePredictionDataLoader

# Initialize with historical data path
loader = CoursePredictionDataLoader(
    historical_data_path=r"C:\AIS_Data_Testing\Historical\2024"
)

# Load and preprocess data
df = await loader.load_training_data(
    start_date="2024-01-01",
    end_date="2024-01-07",
    preprocess=True
)
```

### Advanced Usage
```python
from ml_course_prediction.utils.data_preprocessing import HistoricalDataPreprocessor

# Direct preprocessing
preprocessor = HistoricalDataPreprocessor(
    data_path=r"C:\AIS_Data_Testing\Historical\2024",
    max_speed_knots=50.0  # Custom speed limit
)

# Load and preprocess
df = preprocessor.load_date_range("2024-01-01", "2024-01-31")
df = preprocessor.preprocess_dataframe(df)

# Get summary
summary = preprocessor.get_data_summary(df)
```

## Performance Considerations

### Large Volume Handling
- **7.3M records/day**: Efficient parquet reading
- **Chunked loading**: Optional for very large operations
- **Memory efficient**: Process in date ranges
- **Fast filtering**: Early filtering reduces processing time

### Optimization
- Filter invalid data early
- Use efficient pandas operations
- Process date ranges incrementally
- Optional parallel processing for multiple days

## Next Steps

1. ✅ **Data Loading**: Complete
2. ✅ **Data Preprocessing**: Complete
3. ⏳ **Model Architecture**: Next phase
4. ⏳ **Training**: After model implementation
5. ⏳ **Evaluation**: After training

## Files Modified/Created

### Created
- `utils/data_preprocessing.py` - Historical data preprocessor
- `training/test_data_loading.py` - Test script
- `DATA_TRANSFORMATIONS.md` - Detailed documentation
- `DATA_REVIEW_SUMMARY.md` - This summary

### Updated
- `training/data_loader.py` - Added historical data support
- `utils/__init__.py` - Added preprocessor export
- `models/configs/default_config.yaml` - Added data path

## Testing

Run the test script to verify data loading:
```bash
cd LLM-MLv1/ml_course_prediction
python training/test_data_loading.py
```

## Notes

- All transformations are **configurable** via preprocessor parameters
- Data is **unfiltered** - contains all vessel types and regions
- Preprocessing is **reversible** - original data preserved
- **Memory efficient** - processes large volumes without issues
- **Extensible** - easy to add new transformations

---

**Status**: ✅ Complete  
**Date**: 2025-01-XX  
**Ready for**: Model development and training

