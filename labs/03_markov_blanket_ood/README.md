# Lab 3: Beyond the Markov Blanket

Test MB in OOD setting

## Setup

🤗 [Graphs](https://huggingface.co/datasets/CSE472-blanket-challenge/phase3-graphs) | 🤗 [Datasets](https://huggingface.co/datasets/CSE472-blanket-challenge/phase3-dataset)

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
2. Understand OOD scenarios
   - Covariate Shift
   - Label Shift
3. Compare different feature sets for OOD prediction
   - All features
   - Markov Blanket
   - Parents
   - Children
   - P + C
   - C + Spouses

## Questions

1. Is MB still the optimal feature set for OOD prediction?

## Bonus (5 points, choose one)

1. Test with nonlinear SCMs

## Reading Materials

1. [Do causal predictors generalize better to new domains?](https://proceedings.neurips.cc/paper_files/paper/2024/file/3792ddbf94b68ff4369f510f7a3e1777-Paper-Conference.pdf)
