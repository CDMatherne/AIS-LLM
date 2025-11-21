# ML Course Prediction - Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
cd LLM-MLv1/ml_course_prediction
pip install -r requirements.txt
```

### 2. Verify Backend Access

The data loader integrates with the existing backend system. Ensure you have:
- Access to `backend.data_connector` or `backend.data_cache`
- AIS data available (local files or AWS S3)

### 3. Project Structure

```
ml_course_prediction/
├── data/              # Data storage
├── models/            # Model files
├── training/          # Training scripts
├── inference/         # Prediction interface
├── utils/             # Utilities
├── notebooks/         # Jupyter notebooks
└── tests/             # Unit tests
```

## Development Status

### ✅ Phase 1: Data Pipeline (In Progress)
- [x] Project structure created
- [x] Trajectory processing utilities
- [x] Feature engineering module
- [x] Data loader framework
- [ ] Data validation and quality checks
- [ ] Data exploration notebooks

### ⏳ Phase 2: Model Development (Pending)
- [ ] Model architecture implementation
- [ ] Uncertainty quantification layers
- [ ] Physics constraint integration

### ⏳ Phase 3: Training (Pending)
- [ ] Training script
- [ ] Training loop implementation
- [ ] Model checkpointing

### ⏳ Phase 4: Evaluation (Pending)
- [ ] Evaluation metrics
- [ ] Validation on test set
- [ ] Performance benchmarking

### ⏳ Phase 5: Integration (Pending)
- [ ] Prediction API
- [ ] Integration with existing system
- [ ] Documentation

## Next Steps

1. Complete data pipeline validation
2. Implement model architecture
3. Set up training infrastructure
4. Begin model training on historical data

## Notes

- All development is isolated in `LLM-MLv1` folder
- No modifications to existing project files
- Integration points will be defined in Phase 5

