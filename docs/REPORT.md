# Final Project Report Template

Please submit your final project report follow the structure below.
Note that you only need to largely follow the template;
A lot questions here are only guiding questions to help you think through your report.

The final report should be in pdf format and submit to canvas.

---

- **Name**: Anton Sazonov / Fredo Guan
- **Email**: asazono1@asu.edu / fyguan@asu.edu
- **ASU ID**: <!-- Optional / Last 4 digits -->
- **Team Name**: <!-- Team / repo name -->
- **GitHub Repo**: 
- **Hardware Used**: <!-- e.g., RTX 4090 GPU, Colab T4, CPU only, etc. -->

---

## 1. Problem & Approach Overview

### 1.1 Problem Restatement

- The objective of the project is determining the flow of causality in a network of random variables & using it to predict the target output.
- The two main tasks are the prediction of the Markov blanket and the target variable.
- The performance is measured with the RMSE loss and the jaccard score.

### 1.2 High-Level Approach

The strategy we found to work was to extract a subset of the features and then run the TabPFN classifier on them. This gave us high accuracy predictions of the Markov blanket.

---

## 2. Method & Implementation

### 2.1 Model Design

- Describe the core model or pipeline you implemented:
  - How do you represent each task / dataset?
  - How do you handle different feature dimensions across tasks?
  - How do you combine regression and MB prediction?

### 2.2 Key Techniques and Algorithms

- Explain the key techniques you used

### 2.3 Implementation Details

- Point to the main modules / scripts in your repo:
  - Data loading and task handling
  - Training / meta-training code
  - Inference / evaluation code
- Mention any important implementation tricks (e.g., batching, caching, mixed precision).

---

## 3. Experimental Setup

### 3.1 Datasets and Tasks

- Describe how you used the provided training / validation tasks.
- How did you split data for development (e.g., train/val split across tasks)?

### 3.2 Hyperparameters and Training

- Summarize key hyperparameters (you may also put a detailed table in the Appendix):
  - Learning rates, epochs/steps, batch sizes
  - TabPFN settings (e.g., device, variants used)
  - Any regularization or early-stopping criteria

### 3.3 Compute Resources

- Hardware: GPU(s) vs CPU, memory constraints.
- Approximate training / tuning time.

---

## 4. Results

> Align this section with the official evaluation: RMSE, Jaccard, and the final score.

### 4.1 Main Quantitative Results

- Report your **final leaderboard / submission results**:
  - Final score: $\text{Score} = avg[{\text{RMSE}_i} \times (1 - {\text{Jaccard}_i})]$
- If possible, include a simple table comparing:
  - Baseline vs your final model.

### 4.2 Ablations and Comparisons (Optional)

- Show 1–3 small experiments that justify design choices, e.g.:
  - Different ways to compute MB masks.
  - Different TabPFN variants or hyperparameters.
  - Effect of adding/removing specific components (e.g., embeddings vs no embeddings).
- Present results in small tables or concise plots.

### 4.3 Qualitative Analysis (Optional)

- Any interesting qualitative behavior:
  - Example tasks where MB prediction is especially good or poor.
  - Observations about which features are consistently selected.

---

## 5. Analysis & Insights

> This section is where you can demonstrate **insightful analysis**,
which is also rewarded in the overall project evaluation.

### 5.1 What Worked and Why

- Discuss **why** your approach works (or doesn't) on this benchmark:
  - How does your method exploit the task structure (support/query split)?
  - How does it leverage causal information or MB structure, if at all?

### 5.2 Failure Modes and Limitations

- Describe typical failure cases:
  - Specific types of tasks / environments where performance degrades.
  - Situations where MB predictions are unstable or clearly wrong.
- Discuss possible reasons for these issues.

### 5.3 Future Work

- Suggest 2–3 concrete directions for improvement:
  - Better MB prediction strategies.
  - More principled meta-learning or regularization.
  - Alternative backbone models beyond TabPFN, etc.

---

## 6. Individual Contribution & Reflection

> This section is **personal** and helps justify your individual report score.

### 6.1 Individual Contributions

- Clearly describe what **you** personally did. Examples:
  - Implemented specific modules / training loops.
  - Ran particular experiments or ablations.
  - Designed analysis, visualizations, or debugging tools.
- If work was shared, clarify your role within that shared work.

### 6.2 Learning Outcomes

- What did you learn from this project?
  - About causal inference / Markov Blankets.
  - About meta-learning / tabular foundation models.
  - About practical ML engineering (reproducibility, debugging, scaling).

### 6.3 Challenges and How You Addressed Them

- Describe 1–2 concrete technical or project-management challenges.
- Explain how you tried to solve or mitigate them.

---

## 7. References & Appendix

References:
- Yuan, C., & Malone, B. (2013). Learning optimal Bayesian networks: A shortest path perspective. 



Appendix


- Full tables of hyperparameters.
- Extended ablation results.
- Additional plots (learning curves, per-task metrics).
- Implementation diagrams or pseudocode for complex components.
