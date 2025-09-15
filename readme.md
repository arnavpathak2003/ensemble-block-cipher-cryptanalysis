# Ensemble Learning Approaches for ML-based Cryptanalysis of Block Ciphers

This repository implements ensemble learning techniques (XGBoost, Random Forest, and Gradient Boosting) for machine learning-based cryptanalysis of the SM4 and HIGHT block ciphers. The approach demonstrates superior performance compared to deep learning methods while requiring significantly less computational resources.

## Overview

This work presents a comprehensive evaluation of ensemble learning methods for distinguishing block cipher outputs from random data, a fundamental problem in cryptanalysis. The implementation provides complete workflows for dataset generation, model training, and performance evaluation across multiple cipher configurations.

## Project Structure

```
.
├── hight_final/
│   ├── utils/
│   │   ├── cnn.py                          # CNN implementation for baseline comparison
│   │   ├── hight.py                        # HIGHT cipher implementation (64-bit)
│   │   ├── pkl_preprocessor.py             # Batch data loading and preprocessing
│   │   └── utils.py                        # General utility functions
│   ├── encrypted_output_generator.ipynb    # Generate HIGHT encrypted datasets
│   ├── final_dataprep_into_pkl.ipynb       # Combine and format datasets
│   ├── main.ipynb                          # Fixed-round analysis across deltas
│   ├── main_best_delta.ipynb              # Optimal delta analysis across rounds
│   ├── random_output_generator.ipynb       # Generate random baseline data
│   ├── timer_with_cnn.ipynb               # Performance benchmarking
│   ├── pyproject.toml                      # Project dependencies
│   └── uv.lock                            # Dependency lock file
│
└── sm4_final/
    ├── utils/
    │   ├── cnn.py                          # CNN implementation for baseline comparison
    │   ├── sm4.py                          # SM4 cipher implementation (128-bit)
    │   ├── pkl_preprocessor.py             # Batch data loading and preprocessing
    │   └── utils.py                        # General utility functions
    ├── encrypted_output_generator.ipynb    # Generate SM4 encrypted datasets
    ├── final_dataprep_into_pkl.ipynb       # Combine and format datasets
    ├── main.ipynb                          # Fixed-round analysis across deltas
    ├── main_best_delta.ipynb              # Optimal delta analysis across rounds
    ├── random_output_generator.ipynb       # Generate random baseline data
    ├── timer_with_cnn.ipynb               # Performance benchmarking
    ├── pyproject.toml                      # Project dependencies
    └── uv.lock                            # Dependency lock file
```

## Quick Start

### Prerequisites

Install [uv](https://github.com/astral-sh/uv) for Python package management:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Setup

1. Clone and navigate to the repository:

```bash
git clone <repository-url>
cd ensemble-crypto-analysis
```

2. Choose your cipher directory:

```bash
cd sm4_final    # For SM4 analysis
# or
cd hight_final  # For HIGHT analysis
```

3. Install dependencies:

```bash
uv sync
```

4. Activate environment and start Jupyter:

```bash
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

uv run jupyter lab
```

## Dataset Generation Pipeline

### Step 1: Generate Random Baseline Data

```bash
jupyter notebook random_output_generator.ipynb
```

Creates random binary sequences for the "different bits" class in binary classification. This establishes the baseline for distinguishing cipher outputs from truly random data.

**Key Process:**

- Generates 100,096 random 64-bit (HIGHT) or 128-bit (SM4) sequences
- Creates bit-flipped variants at all possible positions (64 deltas for HIGHT, 128 for SM4)
- Saves structured data for each delta value

### Step 2: Generate Encrypted Outputs

```bash
jupyter notebook encrypted_output_generator.ipynb
```

Produces cipher-specific encrypted data for the "flipped bits" class.

**Configuration:**

- Fixed seed (42) for reproducible master key generation
- Processes plaintext pairs with single-bit differences
- Generates encrypted pairs (C1, C2) for multiple rounds
- HIGHT: Rounds 1-32, 64 delta positions
- SM4: Rounds 1-32, 128 delta positions

### Step 3: Final Data Preparation

```bash
jupyter notebook final_dataprep_into_pkl.ipynb
```

Combines encrypted and random data into balanced training datasets.

**Output:**

- Binary classification datasets (200,192 total samples)
- Perfect class balance (50% flipped_bits, 50% diff_bits)
- Optimized pickle format for efficient batch loading
- Organized by round and delta for systematic analysis

## Analysis Workflows

### Comprehensive Delta Analysis

**`main.ipynb`**: Evaluates all delta values for a fixed number of rounds.

- Tests model performance across the complete delta range
- Identifies optimal input difference positions
- Generates accuracy plots and performance metrics
- Saves results as CSV and pickle files for further analysis

### Round Progression Analysis

**`main_best_delta.ipynb`**: Uses optimal delta values to analyze vulnerability across rounds.

**Optimal Delta Values:**

- **SM4**: Delta = 106 (maximum vulnerability at rounds 4-5)
- **HIGHT**: Delta = 24 (part of arithmetic progression: 8, 24, 40, 56...)

### Performance Benchmarking

**`timer_with_cnn.ipynb`**: Comprehensive comparison of ensemble methods vs. CNN.

**Metrics:**

- Training time per algorithm
- Memory usage patterns
- Accuracy across different data sizes
- Computational efficiency analysis

## Machine Learning Models

### Ensemble Methods

1. **XGBoost**

   - Optimized gradient boosting with L1/L2 regularization
   - Advanced tree pruning and learning rate optimization
   - Incremental training support for large datasets

2. **Random Forest**

   - Bootstrap aggregating with feature randomization
   - Warm-start incremental training
   - Robust against overfitting

3. **Gradient Boosting**
   - Sequential additive modeling
   - Warm-start capability for batch processing
   - Configurable depth and learning rate

### Baseline Comparison

- **Convolutional Neural Network (CNN)**: Deep learning baseline for performance comparison

## Cipher Specifications

### SM4 (Chinese National Standard)

- **Block Size**: 128 bits
- **Key Size**: 128 bits
- **Structure**: Unbalanced Feistel network with 32 rounds
- **S-box**: 8×8 substitution box
- **Dataset**: 128 delta positions, up to 32 rounds analyzed

### HIGHT (Lightweight Block Cipher)

- **Block Size**: 64 bits
- **Key Size**: 128 bits
- **Structure**: Generalized Feistel network (8 branches)
- **Rounds**: 32 total (64 rounds analyzed for research)
- **Dataset**: 64 delta positions, up to 32 rounds analyzed

## Experimental Results

### SM4 Cryptanalysis

- **100% accuracy**: Rounds 1-4 with optimal delta
- **86% accuracy**: Round 5 with delta = 106
- **Vulnerability pattern**: Delta values > 97 show consistent weaknesses
- **Training efficiency**: 2-5x faster than CNN implementations

### HIGHT Cryptanalysis

- **100% accuracy**: Rounds 1-8 with optimal deltas
- **72% accuracy**: Round 9 with delta = 24
- **Vulnerability pattern**: Arithmetic progression (8, 24, 40, 56, 72...)
- **Training efficiency**: 3-6x faster than CNN implementations

### Performance Comparison

- **XGBoost**: Consistently highest accuracy and fastest training
- **Random Forest**: Balanced performance with good interpretability
- **Gradient Boosting**: Strong accuracy but slower than XGBoost
- **CNN**: Competitive accuracy but 3-6x slower training time

## Hardware Requirements

### Minimum Requirements

- **CPU**: Quad-core processor (Intel i5 or equivalent)
- **RAM**: 8GB (16GB recommended for full dataset)
- **Storage**: 10GB free space
- **Python**: 3.11+ with uv package manager

### Recommended Configuration

- **CPU**: Intel Core i7-12700H or equivalent
- **RAM**: 16GB+ for optimal performance
- **GPU**: NVIDIA GPU with CUDA support (for CNN comparisons)
- **Storage**: SSD recommended for faster I/O operations

## Implementation Notes

### Data Format

- **Input**: Binary arrays (64-bit for HIGHT, 128-bit for SM4)
- **Output**: Concatenated ciphertext pairs [C1||C2]
- **Labels**: Binary classification (flipped_bits vs diff_bits)
- **Storage**: Optimized pickle format with batch loading support

### Reproducibility

- Fixed random seeds (42) throughout pipeline
- Deterministic model initialization
- Version-locked dependencies via uv.lock
- Consistent data preprocessing steps

### Scalability

- Incremental training for large datasets
- Batch processing to manage memory usage
- Parallel execution support where applicable
- GPU acceleration available for CNN comparisons

## Usage Examples

### Basic Delta Analysis

```python
from utils.pkl_preprocessor import PickleBatchLoader
import xgboost as xgb

# Load dataset
batch_loader = PickleBatchLoader("dataset.pkl", batch_size=100096)

# Train XGBoost model
model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1)
for X_batch, y_batch in batch_loader.batch_generator():
    model.fit(X_batch, y_batch)

# Evaluate
X_test, y_test = batch_loader.get_test_set()
accuracy = model.score(X_test, y_test)
```

### Performance Timing

```python
import time

def time_algorithm(func, *args):
    start = time.time()
    result = func(*args)
    return result, time.time() - start

accuracy, training_time = time_algorithm(train_xgboost, batch_loader)
print(f"Accuracy: {accuracy:.4f}%, Time: {training_time:.2f}s")
```

## Citation

If you use this code or findings in your research, please cite our work.


