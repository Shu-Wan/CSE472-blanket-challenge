# Lab 2: The Emperor’s New Markov Blankets

Test MB in IID data setting

## Setup

🤗 [Graphs](https://huggingface.co/datasets/CSE472-blanket-challenge/phase1-graphs) | 🤗 [Datasets](https://huggingface.co/datasets/CSE472-blanket-challenge/phase1-dataset)

- Hugging Face Access Token required
- Set up a python environment using uv

```python
uv venv
uv sync --all-groups

uv pip install -e .
```

## Tasks

1. Get familiar with dataset
   - Read the data card and explore the data
   - Load the dataset, explore and visualize one example
2. Understand different feature selection methods
   - CD-MB (e.g., PC)
   - Embedded method (e.g., Lasso)
3. Train a model with selected features
4. Compare results under different settings
   - Different feature selection methods
   - Different models
   - Different SCM types
   - Different data sizes (train data size)

## Questions

1. Is MB truly the upper bound of optimal feature set for prediction?
2. How stable CD-MB merhods are under different data sizes?

## Bonus (5 points, choose one)

1. Graph Structure Analysis

- Test across graphs with varying:
  - **Density**:
  - **Markov Blanket size**: MB size w.r.t total feature size
  - **Graph size**: number of nodes

1. L1 Regularization Tuning

- Implement adaptive alpha selection:
  - Cross-validation grid search
  - Use validation set to choose optimal alpha
  - Information criterion (AIC/BIC) based selection
- Compare with fixed alpha = 0.05
- **Key Question**: Can dynamic alpha selection make L1 competitive with Oracle MB?

## Reading Materials

1. [Chapter 2, Causal Artificial Intelligence](https://causalai-book.net/)
2. <https://sebastianraschka.com/faq/docs/feature_sele_categories.html>
