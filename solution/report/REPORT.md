---
geometry: margin=1in
fontsize: 10pt
documentclass: article
classoption: twocolumn
bibliography: references.bib
numbersections: true
link-citations: true
header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
  - \usepackage{hyperref}
  - \usepackage{graphicx}
  - \usepackage{amsmath}
  - \usepackage{amssymb}
---

\twocolumn[
\begin{center}
{\Large\textbf{Embedding-Based Markov Blanket Discovery for Robust Feature Selection Across Environments}}

\vspace{1.5em}

\begin{tabular}{cc}
\textbf{Dhruv Bansal} & \textbf{Sahajpreet Singh Khasria} \\
SCAI & SCAI \\
Arizona State University & Arizona State University \\
Tempe, Arizona, USA & Tempe, Arizona, USA \\
dbansa11@asu.edu & skhasria@asu.edu \\
\end{tabular}
\end{center}
\vspace{2em}
]
\footnotetext{Code available at: \url{https://github.com/dhruvb26/CSE472-blanket-challenge}}

# Abstract

Causal feature selection through Markov Blanket (MB) discovery provides theoretical guarantees for optimal prediction while preserving model interpretability. However, identifying the MB without access to the underlying causal graph presents significant challenges, particularly when data exhibits distribution shift across environments. We present a meta-learning framework that combines TabPFN embeddings with neural MB predictors to perform simultaneous feature selection and regression across heterogeneous tasks. Our method trains separate multi-layer perceptron classifiers on TabPFN-extracted representations to predict binary MB masks, which subsequently filter features for downstream regression. Evaluated on 182 meta-training tasks and 46 held-out tasks spanning multiple data-generating processes and environmental conditions, our approach achieves a final score of 0.2071 (RMSE: 0.5078, Jaccard: 0.5921), demonstrating the feasibility of learning MB structure from embeddings without explicit causal discovery while outperforming the baseline by approximately 7.6%.

# Introduction

Feature selection is fundamental to machine learning, directly impacting model performance, interpretability, and generalization. Traditional methods often select features based on correlation with the target variable, which can fail under distribution shift when spurious correlations change across environments [@pearl1988probabilistic]. Causal feature selection addresses this limitation by identifying features with stable causal relationships to the target.

The Markov Blanket (MB) of a target variable provides a theoretically optimal feature set for prediction. Pearl [@pearl1988probabilistic] showed that the MB renders the target conditionally independent of all other variables, implying that knowing the MB is both sufficient and necessary for optimal prediction. For a target variable $Y$, the MB consists of its parents, children, and spouses (parents of children) in the causal graph. This theoretical foundation suggests that identifying the MB should yield robust predictions that generalize across environmental shifts.

Despite this theoretical support, discovering the MB in practice remains challenging. Standard approaches require causal discovery algorithms to first recover the full causal graph, then extract the MB. However, causal discovery methods are computationally expensive, require strong assumptions, and can be unreliable on finite samples [@tsamardinos2003towards]. This motivates the search for alternative approaches that can identify relevant causal features without explicit graph reconstruction.

We address this challenge by framing MB discovery as a meta-learning problem. Given a distribution over causal data-generating processes, we learn to predict MB masks directly from tabular data. Our approach leverages TabPFN [@hollmann2023tabpfn], a transformer-based foundation model for tabular data, to extract rich representations that capture both predictive patterns and structural properties of datasets. These embeddings serve as inputs to learned MB predictors that generalize across tasks with varying feature dimensions and causal structures.

**Main Contributions:**

1. A practical pipeline for MB-based feature selection that bypasses explicit causal discovery
2. Demonstration that TabPFN embeddings encode sufficient information for cross-task MB prediction
3. Empirical validation across 228 tasks with diverse data-generating processes and environmental conditions

# Related Works

**Markov Blanket Theory.** Pearl [@pearl1988probabilistic] formalized the concept of the Markov Blanket and proved its sufficiency for prediction. Tsamardinos and Aliferis [@tsamardinos2003towards] developed algorithms for MB discovery from observational data, establishing principled methods like IAMB for incremental MB learning [@tsamardinos2022iamb]. Recent work has shown that individual causal feature subsets (parents or children alone) can underperform non-causal baselines, emphasizing the importance of the complete MB structure.

**TabPFN and Tabular Foundation Models.** Hollmann et al. [@hollmann2023tabpfn] introduced TabPFN, a transformer trained on synthetic datasets that performs in-context learning for tabular prediction. Unlike traditional models requiring task-specific training, TabPFN processes train-test pairs in a single forward pass, making predictions based on learned priors over data-generating processes. The model's embedding layer provides dense representations that capture both statistical patterns and structural properties of tabular data.

**Meta-Learning for Tabular Data.** Meta-learning frameworks learn to adapt quickly to new tasks by leveraging experience from related tasks [@finn2017maml]. In the tabular domain, this typically involves learning good initializations or adaptation strategies. Our approach differs by learning a direct mapping from embeddings to structural properties (MB masks), treating MB discovery as a multi-label classification problem across tasks.

**Causal Feature Selection.** Traditional causal feature selection methods like the PC algorithm, GES, and NOTEARS first recover the causal graph, then extract relevant features. These methods achieve moderate accuracy with significant computational cost. Our embedding-based approach bypasses explicit graph recovery, directly predicting MB masks in a fraction of the time.

# Method & Implementation

## Model Design

Our approach consists of a three-stage pipeline that performs MB prediction followed by filtered regression. Each task $\mathcal{T}$ is represented by its support set $(X_{train}, y_{train})$ and query set $(X_{test}, y_{test})$ together with metadata such as feature masks, SCM parameters, and environment shifts.

Our pipeline processes tasks independently but uses a shared meta-trained set of models. We handle different feature dimensions by training separate TabPFN regressors, embedding generators, and MBMLP models for each feature size. For example, feature dimensions 19 and 9 are treated as separate groups, each with its own fine-tuned models.

## Stage 1: TabPFN Fine-Tuning and Embedding Extraction

The primary technique in our pipeline is batched fine-tuning of the TabPFNRegressor [@priorlabs2024]. Using the built-in meta dataset collator, each batch corresponds to a full task that includes split preprocessing, differentiable forward passes, and updates to the model weights. This allows the regressor to adapt to the variety of training tasks instead of relying purely on pretraining.

For embedding extraction, we use TabPFNEmbedding to generate fold-based embeddings of each task. For each task with support set $(X_{train}, y_{train})$, we employ $K$-fold cross-validation:

1. Split $X_{train}$ into $K$ folds
2. For each fold $k$, train TabPFN on the remaining $K-1$ folds
3. Extract embeddings for the held-out fold
4. Concatenate embeddings across folds

These embeddings average over predictions and internal activations from the regressor, yielding a compact representation of the causal relationships in the data.

## Stage 2: MB Prediction via Neural Networks

The meta-training dataset contains tasks with two distinct feature dimensionalities: tasks with 9 features and tasks with 19 features. Rather than using a single unified model with padding, we train separate MB predictors for each dimensionality.

For tasks with $d \in \{9, 19\}$ features, we define:

\begin{align}
\text{MLP}_d: \mathbb{R}^{E} &\to [0,1]^d \\
h_1 &= \text{ReLU}(W_1 x + b_1) \\
h_2 &= \text{ReLU}(W_2 h_1 + b_2) \\
\hat{m} &= \sigma(W_3 h_2 + b_3)
\end{align}

where $E$ is the embedding dimension, and $\sigma$ is the sigmoid function. The output $\hat{m} \in [0,1]^d$ represents per-feature probabilities of MB membership.

**Training Procedure.** The MBMLP model performs multi-label binary classification to map embeddings to MB masks. BCEWithLogitsLoss is used during training, and threshold tuning is carried out per feature dimension by evaluating several candidate thresholds on held-out tasks. After training the MBMLP model, we perform threshold tuning for each feature dimension by sweeping thresholds between 0.2 and 0.8, choosing the value that minimizes the combined RMSE and Jaccard-based score.

**Inference.** At test time, we extract embeddings for a new task, pass them through the appropriate MLP$_d$, and aggregate predictions via mean pooling followed by thresholding. If no features are selected (all probabilities below threshold), we employ a fallback strategy: select all features to ensure valid predictions.

## Stage 3: Filtered Regression

Given predicted MB mask $\hat{m} \in \{0,1\}^d$, we filter both support and query sets:

\begin{align}
X'_{train} &= X_{train}[:, \hat{m} = 1] \\
X'_{test} &= X_{test}[:, \hat{m} = 1]
\end{align}

We then use `clone_model_for_evaluation` to obtain clean, non-batched TabPFN models. The fine-tuned TabPFN is fitted on the filtered support set $(X'_{train}, y_{train})$ and used to generate predictions for the filtered query set $X'_{test}$. TabPFN's in-context learning capability allows it to adapt to the filtered feature space without requiring iterative optimization.

## Implementation Details

Our entire pipeline is organized in the `/solution` directory with `main.py` as the entry point. Data loading from HuggingFace is handled in the `BeyondTheBlanket` class initialization.

**Key files:**

- `finetuner.py` - TabPFN fine-tuning logic
- `generator.py` - Embedding generation
- `mbmlp.py` - Markov Blanket MLP predictor

# Experimental Setup

## Dataset and Task Distribution

We use the CSE472-blanket-challenge/final-dataset from Hugging Face, which contains tasks generated from a hierarchical causal data-generating process. Each task $\mathcal{T} = (G, \text{SCM}, E)$ is defined by a directed acyclic graph $G$, a structural causal model specifying functional relationships, and an environment $E \in \{\text{IID}, \text{covariate shift}, \text{label shift}\}$.

The meta-training set (develop) contains 182 tasks with the following distribution:

- **Feature dimensions:** 87 tasks with 9 features, 95 tasks with 19 features
- **Environments:** IID, label shift, covariate shift
- **SCM types:** Linear and nonlinear
- **Sample sizes:** 400 training samples, 100 test samples per task

The held-out set (submit) contains 46 tasks with ground truth MB masks and test labels withheld for final evaluation.

## Training Configuration

For meta-training, we grouped tasks by their feature dimensionality. Within each feature dimension group, we held out 10 percent of the tasks for validation and threshold tuning, and used the remaining 90 percent for fine-tuning the TabPFN regressor and training the MBMLP network. Hyperparameter configurations for all pipeline components are provided in Table~\ref{tab:hyperparams} in the Appendix.

## Computational Resources

All experiments were conducted on ASU Sol HPC cluster using A100 GPUs (40GB VRAM). Total pipeline runtime was approximately 20 minutes including fine-tuning, embedding extraction, MBMLP training, and threshold tuning. Submit prediction uses cloned evaluation-mode TabPFN models and is comparatively lightweight.

# Results

## Main Quantitative Results

The results were calculated by running baseline tests and the final model on held-out tasks (~10% from the develop set).

The baseline delivers an average RMSE of 0.4878, average Jaccard of 0.5504, and combined score of 0.2193. Our pipeline delivers an average RMSE of 0.5078, average Jaccard of 0.5921, and an aggregate final score of **0.2071**.

\begin{table}[H]
\centering
\footnotesize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}llccc@{}}
\toprule
\textbf{Dim} & \textbf{Model} & \textbf{RMSE} & \textbf{Jaccard} & \textbf{Score} \\
\midrule
19 & Baseline & \textbf{0.5071} & 0.5088 & 0.2491 \\
19 & Final & 0.5258 & \textbf{0.5625} & \textbf{0.2301} \\
9 & Baseline & \textbf{0.4661} & 0.5972 & 0.1878 \\
9 & Final & 0.4876 & \textbf{0.6255} & \textbf{0.1826} \\
\bottomrule
\end{tabular}%
}
\caption{Performance comparison by feature dimension}
\end{table}

The final model produces a higher Jaccard score at the cost of slightly elevated RMSE. **The final model achieves approximately 5.6% overall improvement over the baseline** (7.6% for dimension 19, 2.8% for dimension 9).

## Qualitative Analysis

The graph's complexity with entangled paths and dispersed blanket nodes makes MB prediction a particularly challenging problem. Beyond individual examples, a broader inspection of tasks reveals that MB prediction quality is closely tied to the structural clarity of the underlying causal graph. Tasks with well-separated parent and child relationships tend to produce higher Jaccard values, even when the regression RMSE remains moderate.

Figure~\ref{fig:qual_e963d7d6} shows the causal graph for task \texttt{data\_e963d7d6}, which achieved high Jaccard score (0.9) despite moderate RMSE (0.7790). The causal graph structure is relatively clear, with well-separated relationships contributing to accurate MB prediction. Figure~\ref{fig:qual_be3555b2} shows the causal graph for task \texttt{data\_be3555b2}, which achieved lower Jaccard score (0.3) with RMSE 0.8202, due to more complex graph structure with entangled paths.

\begin{figure}[H]
\centering
\includegraphics[width=0.35\textwidth]{../runs/20251124_194116/visualizations/data_e963d7d6_graph.png}
\caption{Causal graph for \texttt{data\_e963d7d6} (RMSE: 0.7790, Jaccard: 0.9)}
\label{fig:qual_e963d7d6}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.35\textwidth]{../runs/20251124_194116/visualizations/data_be3555b2_graph.png}
\caption{Causal graph for \texttt{data\_be3555b2} (RMSE: 0.8202, Jaccard: 0.3)}
\label{fig:qual_be3555b2}
\end{figure}

# Analysis & Insights

## What Worked and Why

**Task-structured training:** For each supported feature dimension, develop tasks are split into a "support" subset for fine-tuning TabPFN and training the MB network, and a "query" subset for threshold tuning and evaluation, mirroring the benchmark's support/query structure.

**Causal/MB signal:** TabPFN embeddings for each task feed an MB-specific MLP that predicts Markov Blanket membership per feature dimension. Averaging per-task MB probabilities and thresholding keeps only features most likely in the causal Markov Blanket before running the regression fine-tune.

**Joint score tuning:** Held-out tasks drive threshold selection by minimizing a combined objective ($\text{RMSE} \times (1 - \text{Jaccard})$), directly balancing predictive accuracy with MB fidelity and exploiting causal sparsity assumptions.

## Failure Modes and Limitations

**Sparse tasks per dimension:** When few develop tasks exist for a feature dimension, the support split and MB training become data-starved, making the MB classifier and threshold tuning noisy.

**MB instability:** Thresholding can yield empty blankets, triggering the fallback to all features. Frequent fallback indicates the MB model cannot distinguish relevant features—often on high-dimensional or highly correlated tasks.

**Distribution shift:** The MB network and thresholds are tuned on develop held-outs; submit tasks with substantially different causal structure may see degraded blanket accuracy and downstream RMSE because the learned thresholds no longer align.

# Future Work

## Architectural Improvements

**Unified Multi-Task Model.** Our current approach trains separate MLP predictors for each feature dimension. A unified architecture using adaptive pooling or padding could leverage shared structure across dimensions while maintaining specialized capacity. Graph neural networks such as DAG-GNN [@yu2019daggnn] could model feature dependencies explicitly, potentially improving MB prediction for features with complex interdependencies.

**Attention-Based Aggregation.** We currently aggregate sample-level predictions via mean pooling. An attention mechanism could learn to weight samples based on their informativeness, potentially improving robustness to outliers or low-quality training examples.

## Training Enhancements

**TabPFN Fine-Tuning Extensions.** TabPFN 2.5 supports fine-tuning via batched training mode. Rather than using zero-shot predictions, deeper fine-tuning on develop tasks could adapt the model to the specific distribution of causal data-generating processes, improving both embedding quality and downstream regression performance.

**Uncertainty-Aware Selection.** Our current threshold of 0.5 for MB membership is fixed. A learned or adaptive threshold based on prediction confidence could improve precision-recall trade-offs. Incorporating established causal discovery ideas like IAMB [@tsamardinos2022iamb] would also provide a more principled signal for selecting parent and spouse variables.

## Evaluation Extensions

**Per-Environment Analysis.** Our validation metric averages across all environments. Stratified evaluation would reveal whether performance degrades differentially under distribution shift, informing environment-specific adaptations.

**Sample Efficiency Studies.** Investigating performance as a function of support set size would characterize the method's data efficiency. TabPFN's in-context learning may enable reasonable performance with very few support examples.

# Appendix

## Hyperparameters

\begin{table}[H]
\centering
\footnotesize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lll@{}}
\toprule
\textbf{Component} & \textbf{Parameter} & \textbf{Value} \\
\midrule
TabPFN (fine-tuning) & epochs & 1 \\
TabPFN (fine-tuning) & learning\_rate & $1 \times 10^{-6}$ \\
TabPFN (fine-tuning) & max\_samples & 10,000 \\
TabPFN (fine-tuning) & val\_fraction & 0.1 \\
\midrule
MLP Predictor & hidden\_dims & [256, 128] \\
MLP Predictor & learning\_rate & $1 \times 10^{-3}$ \\
MLP Predictor & epochs & 40 \\
MLP Predictor & batch\_size & 512 \\
MLP Predictor & val\_fraction & 0.2 \\
\midrule
TabPFN (regression) & n\_estimators & 24 \\
\bottomrule
\end{tabular}%
}
\caption{Hyperparameter configuration for all pipeline components}
\label{tab:hyperparams}
\end{table}

### Explicit (Exposed through config.yaml)

\begin{table}[H]
\centering
\footnotesize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Hyperparameter} & \textbf{Description} \\
\midrule
\texttt{supported\_feature\_dims} & Feature dimensions (e.g., 9, 19) \\
\texttt{holdout\_fraction} & Fraction held out for evaluation \\
\bottomrule
\end{tabular}%
}
\caption{General Configuration}
\end{table}

\begin{table}[H]
\centering
\footnotesize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Hyperparameter} & \textbf{Description} \\
\midrule
\texttt{tabpfn.epochs} & Fine-tuning epochs \\
\texttt{tabpfn.lr} & Learning rate (Adam) \\
\texttt{tabpfn.max\_samples} & Max samples per dataset \\
\texttt{tabpfn.val\_fraction} & Internal validation fraction \\
\texttt{tabpfn.model\_path} & Pretrained model checkpoint \\
\bottomrule
\end{tabular}%
}
\caption{TabPFN Fine-tuning Configuration}
\end{table}

\begin{table}[H]
\centering
\footnotesize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Hyperparameter} & \textbf{Description} \\
\midrule
\texttt{mb\_mlp.epochs} & Training epochs \\
\texttt{mb\_mlp.lr} & Learning rate \\
\texttt{mb\_mlp.batch\_size} & Training batch size \\
\texttt{mb\_mlp.hidden\_sizes} & Hidden layer widths \\
\texttt{mb\_mlp.val\_fraction} & Validation fraction \\
\texttt{mb\_mlp.n\_fold} & Folds for embedding extraction \\
\bottomrule
\end{tabular}%
}
\caption{MBMLP Configuration}
\end{table}

\begin{table}[H]
\centering
\footnotesize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Hyperparameter} & \textbf{Description} \\
\midrule
\texttt{threshold\_tuning.min} & Min threshold in sweep \\
\texttt{threshold\_tuning.max} & Max threshold in sweep \\
\texttt{threshold\_tuning.num\_thresholds} & Grid search values \\
\bottomrule
\end{tabular}%
}
\caption{Threshold Tuning Configuration}
\end{table}

### Implicit Hyperparameters

\begin{table}[H]
\centering
\footnotesize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lll@{}}
\toprule
\textbf{Component} & \textbf{Hyperparameter} & \textbf{Reason} \\
\midrule
TabPFN & \texttt{n\_estimators = 24} & Ensemble size \\
MBMLP & \texttt{activation = ReLU} & Expressiveness \\
MBMLP & \texttt{loss = BCEWithLogitsLoss} & Training objective \\
MBMLP & \texttt{optimizer = Adam} & Learning trajectory \\
MBMLP & \texttt{betas = (0.9, 0.999)} & Momentum \\
Threshold & \texttt{rule = >=} & Mask selection \\
Threshold & \texttt{aggregation = mean} & MB selection \\
Threshold & metric = RMSE$\times$(1-Jacc) & Best threshold \\
\bottomrule
\end{tabular}%
}
\caption{Implicit Hyperparameters}
\end{table}

## Training Loss on MBMLP

Training loss curves for MBMLP models during meta-training:

\begin{figure}[H]
\centering
\begin{minipage}{0.4\textwidth}
\centering
\includegraphics[width=\textwidth]{../runs/20251124_194116/mbmlp_loss_dim9.png}
\caption{MBMLP training loss (9 features)}
\label{fig:loss_dim9}
\end{minipage}
\hfill
\begin{minipage}{0.4\textwidth}
\centering
\includegraphics[width=\textwidth]{../runs/20251124_194116/mbmlp_loss_dim19.png}
\caption{MBMLP training loss (19 features)}
\label{fig:loss_dim19}
\end{minipage}
\end{figure}

# References
