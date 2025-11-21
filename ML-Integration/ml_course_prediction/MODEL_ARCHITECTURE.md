# Model Architecture - Hybrid LSTM-Transformer

## Overview

The model uses a hybrid architecture combining:
- **LSTM Encoder**: Captures short-term temporal patterns (2-8 points)
- **Transformer Encoder**: Captures long-term dependencies via self-attention
- **Uncertainty Head**: Provides 1σ confidence intervals for predictions

## Architecture Components

### 1. LSTM Encoder (`lstm_encoder.py`)
- Processes variable-length sequences (2-8 points)
- Handles padding and masking for efficient batching
- Extracts temporal features from short-term patterns
- Configurable: layers, hidden size, bidirectional, dropout

### 2. Transformer Encoder (`transformer_encoder.py`)
- Uses self-attention to model relationships between all time steps
- Positional encoding for temporal order
- Handles variable-length sequences with padding masks
- Configurable: layers, heads, feedforward size, dropout

### 3. Uncertainty Head (`uncertainty_head.py`)
- Predicts both mean and variance for each output dimension
- Provides 1σ confidence intervals (68% coverage target)
- Supports sampling from predicted distribution
- Uses log-variance for numerical stability

### 4. Hybrid Model (`hybrid_model.py`)
- Combines LSTM and Transformer outputs via feature fusion
- Predicts:
  - **Position** (LAT, LON) with uncertainty
  - **Speed** (SOG) - optional
  - **Course** (COG) - optional
- Handles variable-length input sequences (2-8 points)

## Model Specifications

### Input
- **Shape**: `(batch_size, seq_len, input_size)`
- **seq_len**: 2-8 points (AIS reports every 3 hours)
- **input_size**: Number of features (LAT, LON, SOG, COG, Heading, etc.)

### Output
- **Position**: Dict with `mean`, `std`, `var`, `lower_68`, `upper_68`
- **Speed**: Predicted SOG (knots)
- **Course**: Predicted COG (degrees)

### Default Configuration
- LSTM: 2 layers, 128 hidden units, dropout 0.2
- Transformer: 4 layers, 256 hidden, 8 heads, 1024 feedforward
- Fusion: 256 hidden units
- Prediction horizon: 48 hours

## Usage

### Create Model from Config
```python
from models.model_factory import create_model, load_config

# Load configuration
config = load_config('models/configs/default_config.yaml')

# Create model
model = create_model(config=config, input_size=5)  # 5 features
```

### Forward Pass
```python
import torch

# Input: (batch_size, seq_len, input_size)
x = torch.randn(4, 5, 5)  # 4 sequences, 5 points each, 5 features

# Forward pass
outputs = model(x)

# Access predictions
position_mean = outputs['position']['mean']  # (4, 2) - LAT, LON
position_std = outputs['position']['std']   # (4, 2) - uncertainty
speed = outputs['speed']                     # (4, 1) - SOG
course = outputs['course']                   # (4, 1) - COG
```

### Variable-Length Sequences
```python
# Sequences with different lengths
x = torch.randn(4, 8, 5)  # Pad to max length
lengths = torch.tensor([5, 6, 4, 7])  # Actual lengths

outputs = model(x, lengths=lengths)
```

### Prediction with Uncertainty
```python
# Get predictions with confidence intervals
predictions = model.predict(x, return_uncertainty=True)

# Access 68% confidence interval
lower = predictions['position']['lower_68']
upper = predictions['position']['upper_68']
```

### Save/Load Model
```python
from models.model_factory import save_model, load_model

# Save
save_model(model, 'models/trained/model.pth', config=config, input_size=5)

# Load
model = load_model('models/trained/model.pth', config_path='models/configs/default_config.yaml')
```

## Testing

Run the test script to verify the model works:
```bash
python LLM-MLv1/ml_course_prediction/models/test_model.py
```

This tests:
- Model creation from config
- Forward pass with fixed and variable-length sequences
- Prediction method with uncertainty
- Uncertainty interval calculation

## Model Size

Typical model has:
- **Total parameters**: ~500K - 1M (depending on configuration)
- **Trainable parameters**: All parameters are trainable
- **Memory**: ~10-20 MB (without optimizer state)

## Next Steps

1. **Training Infrastructure**: Create training loop, loss functions, optimizers
2. **Data Integration**: Connect model to data loader for training
3. **Loss Functions**: Implement multi-objective loss (position, speed, course, uncertainty, physics)
4. **Evaluation**: Create evaluation metrics and validation scripts

## Key Features

✅ **Variable-length sequences**: Handles 2-8 points per 24-hour window  
✅ **Uncertainty quantification**: Provides 1σ confidence intervals  
✅ **Hybrid architecture**: Combines LSTM and Transformer strengths  
✅ **Physics-aware**: Ready for physics constraint integration  
✅ **Configurable**: All parameters via YAML config  
✅ **Production-ready**: Save/load, batch processing, GPU support  

