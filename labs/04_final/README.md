# Final Project: Enter the Challenge

The true challenge awaits.

In the final project, you will put what you've learned to build a model that
performs causal feature selection across environments without sacrificing predictive performance.

## Problem Formulation

We formulate our objective as a **meta-learning problem** where the goal is to learn a prior over
causal data generating processes to perform two downstream tasks: regression and Markov Blanket
(MB) discovery.

### Task Distribution and Data Generation Processes

In our setting, a **task** $\mathcal{T}$ corresponds to a specific dataset generated from a causal
mechanism. The distribution of tasks $p(\mathcal{T})$ is defined by a hierarchical generative
process:

1. **Causal Graph ($G$)**: A Directed Acyclic Graph (DAG) is sampled, defining the structural
   dependencies between variables.
2. **Structural Causal Model (SCM)**: Functional relationships and noise distributions are
   assigned to the edges and nodes of $G$.
3. **Environment ($E$)**: An environment defines the specific interventional or observational
   distribution (e.g., IID, covariate shift, or label shift).

Thus, a task is fully determined by the tuple $\mathcal{T} = (G, \text{SCM}, E)$. Sampling from
this task produces a dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, where $x_i \in \mathbb{R}^d$
are features and $y_i \in \mathbb{R}$ is the target variable.

See `src/blanket/datasets` on how data are generated

### A Model for Bayesian Prediction

We treat inference as a Bayesian prediction problem.

**Training**:

During meta-training, we have access to a set of tasks $\{\mathcal{T}_j\}_{j=1}^M$. For each task,
we observe:

- The dataset $\mathcal{D}_j = (X^{(j)}, y^{(j)})$.
- Meta-information of the dataset, specifically the **Markov Blanket (MB)** mask
  $m^{(j)} \in \{0, 1\}^d$, type of SCM (linear, nonlinear),
  Environment (IID, Covariate shift, Label shift).

**Testing (Inference)**:

At test time, we are presented with a new task $\mathcal{T}_{new}$ (unknown $G$ and SCM)
represented by a support set of $k$ examples (a "few-shot" context):
$$ \mathcal{D}_{support} = \{(x_i, y_i)\}_{i=1}^k $$

Our goal is to predict two outcomes given this support set and a query point $x_*$:

1. **Regression**: Predict the target posterior $p(y_* \mid x_*, \mathcal{D}_{support})$.
2. **Feature Selection**: Predict the Markov Blanket mask $p(m \mid \mathcal{D}_{support})$.

## 2. TabPFN Capabilities

### 2.1 Core Features

From [TabPFN 2.5 documentation](https://docs.priorlabs.ai/):

**Specifications**:

- Max samples: 50,000 (TabPFN 2.5)
- Max features: 2,000
- GPU recommended (CPU feasible for <1K samples)
- Handles missing values natively
- Zero-shot learning (no hyperparameter tuning)

**Model Variants**:

- **TabPFN-2.5**: Trained on synthetic data only
- **Real-TabPFN-2.5**: Fine-tuned on real-world data (better empirical performance)

### 2.2 Regression API

```python
from tabpfn import TabPFNRegressor

model = TabPFNRegressor(device="cuda")
model.fit(X_train, y_train)

# Point predictions
y_pred = model.predict(X_test, output_type="mean")

# Uncertainty quantification
quantiles = model.predict(X_test, output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
```

**Outputs**:

1. **Point predictions**: mean, median, mode
2. **Quantile regression**: Prediction intervals
3. **Full distribution**: Raw logits for custom losses

### 2.3 Embedding Extraction

From [embeddings documentation](https://docs.priorlabs.ai/capabilities/embeddings):

```python
from tabpfn_extensions.embedding import TabPFNEmbedding

# Cross-validated embeddings (more robust)
embedding_extractor = TabPFNEmbedding(tabpfn_regressor, n_fold=5)
embedding_extractor.fit(X_train, y_train)
embeddings = embedding_extractor.get_embeddings(X_train, y_train, X_test)
# Shape: (n_samples, embedding_dim)
```

**Key Property**: Recent research ([arXiv:2511.07236](https://arxiv.org/abs/2511.07236))
shows TabPFN embeddings encode causal structure, making them suitable for MB prediction.

For more details on TabPFN, refer to the [resources section](#resources) below.

## Possible Solution

TabPFN is not natively designed for MB prediction.
Below I provide sketches of several strategies to tackle the problem.

### Solution 1: Embedding-Based MB Predictor

Extract [TabPFN embeddings](https://docs.priorlabs.ai/capabilities/embeddings) as dataset
representations, use them to train (fine-tune) a classifier to predict MB.
During meta-testing, extract embeddings from the support set and predict MB.

- Stage 1: Extract Embeddings
- Stage 2: Train MB Predictor

Challenges:

- How to handle different feature dimensions across datasets?
- How to fine-tune effectively with limited data? Fine-tune each dataset separately or jointly?

### Solution 2: Per-Feature Embedding Predictor

Instead of dataset-level, create per-feature representations.

Pros and Cons:

- In addition to Solution 1, train separate TabPFN for separate features
- No feature alignment issues

### Solution 3: Wrapper-Based MB Search (or HPO with `optuna`)

Use TabPFN as oracle to score feature subsets directly.

**Approach**:

```python
def score_feature_subset(mask, X_train, y_train, X_val, y_val):
    X_train_masked = X_train[:, mask == 1]
    X_val_masked = X_val[:, mask == 1]

    model = TabPFNRegressor()
    model.fit(X_train_masked, y_train)
    y_pred = model.predict(X_val_masked)

    return -mean_absolute_error(y_val, y_pred)  # Negative MAE (higher is better)

# Optimize via greedy forward selection, Bayesian optimization, or genetic algorithm
from sklearn.feature_selection import SequentialFeatureSelector
selector = SequentialFeatureSelector(TabPFNRegressor(), n_features_to_select="auto", direction="forward")
selector.fit(X_train, y_train)
selected_mask = selector.get_support()
```

Pros and Cons:

- Computationally expensive (multiple TabPFN fits)
- No training required, use TabPFN as black-box oracle

### Solution 4: SHAP-Based Feature Importance

Use TabPFN's interpretability tools to compute feature importance, threshold for MB.

**Approach**:

```python
from tabpfn_extensions.interpretability import TabPFNFeatureImportance
import shap

model = TabPFNRegressor()
model.fit(X_train, y_train)

# Compute SHAP values
explainer = shap.Explainer(model.predict, X_train)
shap_values = explainer(X_test)
feature_importance = np.abs(shap_values.values).mean(axis=0)

# Threshold to binary mask
k = expected_mb_size  # Or use percentile
mask = np.zeros(d)
mask[np.argsort[feature_importance](-k:)] = 1
```

Pros and Cons:

- No training required
- Arbitrary thresholding may affect MB accuracy

## Submissions

- Create a fork from main
- Create a PR (before due date, will be evaluated on the last commit before due date)
- Specify required hardware (GPU/CPU) in the PR description
- If use Colab, upload the notebook and link it in the PR description

## Evaluation

We evaluate the model on a set of $N$ unseen testing tasks $\{\mathcal{T}_j\}_{j=1}^N$.
For each task $\mathcal{T}_j$, the dataset is split into a support set and a query set:

- **Support Set**: $\mathcal{D}^{(j)}_{support} = \{(x_i, y_i)\}_{i=1}^k$, used for in-context
  learning or fine-tuning.
- **Query Set**: $\mathcal{D}^{(j)}_{query} = \{(x_i, y_i)\}_{i=1}^{N_{query}}$, used for
  computing the regression metric.

For each task, the model predicts:

1. The target values $\hat{y}$ for samples in $\mathcal{D}^{(j)}_{query}$.
2. The Markov Blanket mask $\hat{m}^{(j)} \in \{0, 1\}^d$.

### Metrics

The performance is measured using two metrics:

1. **Root Mean Squared Error (RMSE)** (Lower is better):
   $$ \text{RMSE}_j = \sqrt{\frac{1}{N_{query}} \sum_{(x, y) \in \mathcal{D}^{(j)}_{query}}
   (y - \hat{y}(x))^2} $$
   We report the average RMSE over all $N$ tasks:
   $$ \overline{\text{RMSE}} = \frac{1}{N} \sum_{j=1}^N \text{RMSE}_j $$

2. **Jaccard Score** (Higher is better):
   Measures the overlap between the predicted Markov Blanket $\hat{m}^{(j)}$ and the true
   Markov Blanket $m^{(j)}_{true}$.
   $$ \text{Jaccard}_j = \frac{|\hat{m}^{(j)} \cap m^{(j)}_{true}|}{|\hat{m}^{(j)}
   \cup m^{(j)}_{true}|} $$
   We report the average Jaccard score over all $N$ tasks:
   $$ \overline{\text{Jaccard}} = \frac{1}{N} \sum_{j=1}^N \text{Jaccard}_j $$

### Final Score

The final score combines regression performance and feature selection accuracy.

$$ \text{Score} = \overline{\text{RMSE}} \times (1 - \overline{\text{Jaccard}}) $$

**Lower score is better.**

### Grading

The final project points will be rewarded based on the rankings of the final score.

- **1st place**: 30 points
- **Last place**: 10 points
- **Others**: Min-max scaling between 10 and 30 points:

$$
\text{Points} = 10 + 20 \times \frac{\text{Score}_{last} - \text{Score}^*}{\text{Score}_{last} -
\text{Score}_{first}}
$$

## Implementation

Example: [final.ipynb](final.ipynb)
Hardware: Single A100 on Sol

Environment setup:

```bash
uv sync --all-groups

uv pip install -e .
```

Environement variables (if needed):

```bash
export TABPFN_CACHE_DIR=/path/to/tabpfn_cache
export TABPFN_DISABLE_TELEMETRY=1
export PYTORCH_ALLOC_CONF=max_split_size_mb:512
```

## Resources

### TabPFN Documentation

- **Official Docs**: <https://docs.priorlabs.ai/>
- **GitHub**: <https://github.com/PriorLabs/TabPFN>
- **Hugging Face**: <https://huggingface.co/Prior-Labs/tabpfn_2_5>
- **Extensions**: <https://github.com/PriorLabs/tabpfn-extensions>
- **Regression**: <https://docs.priorlabs.ai/capabilities/regression>
- **Embeddings**: <https://docs.priorlabs.ai/capabilities/embeddings>

### Papers

1. Grinsztajn et al., "Advancing the State of the Art in Tabular Foundation Models"
2. Hollmann et al., "Accurate predictions on small data with a tabular
foundation model", doi:10.1038/s41586-024-08328-6
3. "Does TabPFN Understand Causal Structures?", arXiv:2511.07236
4. Nastl & Hardt, "Do causal predictors generalize better to new
domains?", NeurIPS
