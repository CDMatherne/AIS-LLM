# ML Course Prediction System

This module implements machine learning models to predict future ship courses up to 48 hours after the last known position, enabling law enforcement interdiction of anomalous vessels.

## Project Structure

```
ml_course_prediction/
├── data/              # Data storage (raw, processed, features)
├── models/            # Model architectures, trained models, configs
├── training/          # Training scripts and infrastructure
├── inference/         # Prediction interface and API integration
├── utils/             # Utility functions (features, trajectories, visualization)
├── notebooks/         # Jupyter notebooks for exploration and development
└── tests/             # Unit tests

```

## Key Features

- **48-Hour Prediction**: Forecasts vessel positions up to 48 hours ahead
- **Uncertainty Quantification**: Provides confidence intervals (1σ) for predictions
- **Anomaly Handling**: Works with vessels exhibiting suspicious behavior
- **Physics Constraints**: Incorporates maritime physics (max speed, turn radius)
- **Hybrid Architecture**: LSTM-Transformer model for optimal performance

## Quick Start

See the main plan document: `ML_COURSE_PREDICTION_PLAN.md`

## Development Status

- Phase 1: Data Pipeline (In Progress)
- Phase 2: Model Development (Pending)
- Phase 3: Training & Tuning (Pending)
- Phase 4: Evaluation (Pending)
- Phase 5: Integration (Pending)

