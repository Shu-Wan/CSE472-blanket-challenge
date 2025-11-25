# CSE 472 Final Project: Group 1
Sameera Shah, Tanmayi Ghadge

**Final Solution + Baseline + Previous Attempt**

This repository contains the final implementation of our solution to the **Blanket Challenge**.

The `final.ipynb` script includes:
- Final solution
- Baseline Solution
- A previous attempted approach using TabPFN embeddings

---

## Approach Overview

### Final Model
1. **Markov Blanket Prediction**
   - Compute per-feature statistics (correlation, predictive strength, variance).
   - Train an **MLP classifier** for per-feature binary MB prediction.
   - Apply only on support set (no information leakage).

2. **Feature-Based Regression**
   - Select features masked by predicted MB.
   - Use `TabPFNRegressor` for final predictions.
   - Use **support/query split** for meta-testing.

 Final develop score: `0.2841` 
RMSE: 0.5188 | Jaccard: 0.4600

---

## Performance Comparison

| Model | Avg RMSE | Avg Jaccard | Final Score |
|-------|----------|-------------|-------------|
| **Final Solution** | 0.5188 | 0.4600 |  **0.2841** |
| **Baseline** (No MB) | **0.5115** | 0.0000 |  0.5115 |
| Old Attempt (TabPFN Embeddings) | 0.5846 | 0.4415 | 0.3265 |

Final approach is significantly better due to improved causal feature selection.

---

## Repository Structure

```
CSE472-Blanket-Challenge/
│
└── labs
│
└── final_project_implementation/
    └── final.ipynb
    └── submission.csv 
    └── README.MD 
    └── requirements.txt
```

---

## Hugging Face Authentication (Required)

TabPFN model access is gated.
1. Get access to TabPFN Model here: https://huggingface.co/Prior-Labs/tabpfn_2_5
2. Get a token  
    https://huggingface.co/settings/tokens

3. Authenticate inside notebook/script:

```python
from huggingface_hub import login
login()
```

---

## Running the Code

### Run the final.ipynb cell by cell, all requirements are taken care of within the cell.

```
Submission CSV

The generated CSV is in the final_project_implementation folder as submission.csv

