# Data Transformations for Historical AIS Data

## Data Source

**Location**: `C:\AIS_Data_Testing\Historical\2024`  
**Format**: Parquet files (one per day)  
**Naming**: `AIS_YYYY_MM_DD.parquet`  
**Volume**: ~366 files (full year 2024), ~7.3M records per day

## Data Structure

### Columns Available
- **MMSI**: Vessel identifier (int64)
- **BaseDateTime**: Timestamp (object/string, needs conversion)
- **LAT**: Latitude (float64)
- **LON**: Longitude (float64)
- **SOG**: Speed Over Ground (float64)
- **COG**: Course Over Ground (float64)
- **Heading**: Vessel heading (float64)
- **VesselName**: Vessel name (object, optional)
- **IMO**: IMO number (object, optional)
- **CallSign**: Call sign (object, optional)
- **VesselType**: Vessel type code (float64)
- **Status**: Navigation status (float64)
- **Length**: Vessel length (float64)
- **Width**: Vessel width (float64)
- **Draft**: Vessel draft (float64)
- **Cargo**: Cargo type (float64)
- **TransceiverClass**: AIS class (object)

### Data Quality Observations
- **BaseDateTime**: Stored as string, needs conversion to datetime
- **Missing Values**: 
  - VesselName: ~0.1% missing
  - IMO: ~31% missing
  - CallSign: ~11% missing
  - VesselType: ~0.04% missing
  - Status: ~28% missing
  - Length/Width/Draft: Some missing
- **Invalid Values**:
  - SOG: ~0.2% have values > 100 knots (need filtering)
  - All other numeric fields within valid ranges

## Required Transformations

### 1. DateTime Conversion
**Issue**: BaseDateTime stored as string/object  
**Solution**: Convert to pandas datetime
```python
df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], errors='coerce')
```

### 2. Position Validation
**Filters Applied**:
- Latitude: -90 to 90 degrees
- Longitude: -180 to 180 degrees
- Remove zero positions (0, 0) - indicates missing data

### 3. Speed Validation
**Filters Applied**:
- SOG: 0 to 100 knots (configurable, default max: 100)
- NaN values allowed (stationary vessels)
- Invalid speeds removed

### 4. Course Validation
**Filters Applied**:
- COG: 0 to 360 degrees
- NaN values allowed (valid for stationary vessels)

### 5. Missing Value Handling
**Strategy**:
- **VesselType**: Fill with 0 (unknown) if missing
- **Length/Width/Draft**: Fill with 0 if missing (filtered later if needed)
- **SOG/COG/Heading**: Allow NaN (valid for stationary vessels)
- **VesselName/IMO/CallSign**: Keep as NaN (optional fields)

### 6. Data Sorting
**Required**: Sort by MMSI and BaseDateTime for trajectory processing
```python
df = df.sort_values(['MMSI', 'BaseDateTime']).reset_index(drop=True)
```

## Preprocessing Pipeline

### Step 1: Load Data
- Load parquet files for date range
- Combine multiple days into single DataFrame

### Step 2: Convert Types
- Convert BaseDateTime to datetime
- Ensure numeric types are correct

### Step 3: Validate and Filter
- Remove invalid positions
- Remove invalid speeds
- Remove invalid courses
- Remove rows with invalid timestamps

### Step 4: Handle Missing Values
- Fill optional fields appropriately
- Preserve NaN where valid (SOG, COG for stationary)

### Step 5: Sort and Index
- Sort by MMSI and time
- Reset index

## Performance Considerations

### Large Volume Handling
- **7.3M records per day**: Need efficient processing
- **Chunked loading**: Optional chunk_size parameter for very large files
- **Memory management**: Process in batches if needed

### Optimization Strategies
1. **Filter early**: Remove invalid data before expensive operations
2. **Use categorical types**: For MMSI, VesselType (if appropriate)
3. **Lazy evaluation**: Process only required date ranges
4. **Parallel processing**: Can process multiple days in parallel

## Usage Example

```python
from ml_course_prediction.utils.data_preprocessing import HistoricalDataPreprocessor
from ml_course_prediction.training.data_loader import CoursePredictionDataLoader

# Initialize preprocessor
preprocessor = HistoricalDataPreprocessor(
    data_path=r"C:\AIS_Data_Testing\Historical\2024"
)

# Load and preprocess single day
df = preprocessor.load_date_file("2024_01_01")
df = preprocessor.preprocess_dataframe(df)

# Or use data loader
loader = CoursePredictionDataLoader(
    historical_data_path=r"C:\AIS_Data_Testing\Historical\2024"
)

# Load date range with preprocessing
df = await loader.load_training_data(
    start_date="2024-01-01",
    end_date="2024-01-07",
    preprocess=True
)
```

## Expected Data Quality After Preprocessing

- **Valid positions**: All LAT/LON within valid ranges
- **Valid speeds**: All SOG within 0-100 knots (or NaN)
- **Valid courses**: All COG within 0-360 degrees (or NaN)
- **Sorted**: Data sorted by MMSI and time
- **Typed**: BaseDateTime as datetime, numeric types correct
- **Clean**: Invalid records removed (~0.2-0.5% typically)

## Notes

- Data is **unfiltered** - contains all vessel types and all regions
- Good for training diverse models
- Can filter by vessel type or region after loading
- Preprocessing is configurable (speed limits, etc.)

