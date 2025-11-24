# Beyond the Blanket - Solution

This folder contains the solution implementation for the CSE472 Blanket Challenge for Group 7, which tackles **Causal Feature Selection** and **Markov Blanket Discovery** using a hybrid approach combining TabPFN fine-tuning with neural network-based feature selection.

The solution implements a three-stage pipeline:

1. **TabPFN Fine-tuning**: Fine-tune a pre-trained TabPFN (Tabular Prior-Fitted Network) regressor on domain-specific tasks
2. **Embedding Generation**: Extract TabPFN embeddings for Markov blanket prediction
3. **Markov Blanket MLP**: Train a neural network to predict which features belong in the Markov blanket
4. **Threshold Tuning**: Optimize the classification threshold for feature selection
5. **Final Prediction**: Make predictions on the submission dataset using the trained models

<div align="center">
<img src="assets/pipeline.png" alt="Pipeline Diagram" width="400">
</div>

## Files

### Core Pipeline Files

| File               | Purpose                                                              |
| ------------------ | -------------------------------------------------------------------- |
| **`main.py`**      | Main pipeline orchestrator implementing the `BeyondTheBlanket` class |
| **`finetuner.py`** | TabPFN fine-tuning logic with validation tracking                    |
| **`generator.py`** | TabPFN embedding generation for training and testing                 |
| **`mbmlp.py`**     | Markov Blanket MLP model for feature mask prediction                 |
| **`config.yaml`**  | Hyperparameters and configuration settings                           |

### Utility Files

| File               | Purpose                                                              |
| ------------------ | -------------------------------------------------------------------- |
| **`util.py`**      | Utility functions for run management, file I/O, and results tracking |
| **`baseline.py`**  | Baseline evaluation using vanilla TabPFN without feature selection   |
| **`visualize.py`** | Visualization tools for analyzing individual task results            |

### Generated Outputs

- **`runs/`**: Directory containing timestamped run outputs
  - `results.json`: Metrics and evaluation results
  - `run_info.json`: Configuration snapshot for reproducibility
  - `submission.csv`: Final predictions for submission
  - `mbmlp_loss_dim*.png`: Training/validation loss curves
  - `visualizations/`: Optional task-specific diagnostic plots

---

## Installation

### Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Key Dependencies

- **`tabpfn`**: TabPFN regressor for meta-learning
- **`tabpfn-extensions`**: Embedding extraction utilities
- **`torch`**: Neural network training
- **`datasets`**: Hugging Face datasets integration
- **`blanket`**: Custom metrics and utilities (from parent directory)
- **`numpy`**, **`matplotlib`**, **`tqdm`**: Standard scientific computing

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Optional: Set custom paths
export HF_HOME=/path/to/huggingface/cache
```

---

## Usage

### Running the Full Pipeline

Execute the main pipeline:

```bash
python main.py
```

This will:

1. Load the develop and submit datasets from Hugging Face
2. Fine-tune TabPFN for each feature dimension
3. Train Markov Blanket MLPs
4. Tune thresholds on holdout data
5. Generate predictions for submission
6. Save all results to a timestamped `runs/` directory

### Evaluating the Baseline

Compare against a vanilla TabPFN baseline:

```bash
python baseline.py --config config.yaml
```

This runs TabPFN without feature selection and reports:

- Average RMSE
- Average Jaccard score (all features selected)
- Combined score

### Visualizing Results

Inspect a specific task from a run:

```bash
python visualize.py --run runs/20251123_154729 --data-id data_be3555b2
```

Generates:

- Causal graph visualization
- True vs. predicted Markov blanket comparison
- Residual plots for regression

---
