# GG

The journey comes to an end. Congrats to all teams to reach this far!

## :trophy: Leaderboard

## Team Roster

| Group | Members | Repo | Submission File |
|-------|---------|------|-----------------|
| 1 | Sameera, Tanmayi | [sameerashahh/CSE472-blanket-challenge](https://github.com/sameerashahh/CSE472-blanket-challenge/tree/project2_implementation) | [`final_project_implementation/submission.csv`](https://raw.githubusercontent.com/sameerashahh/CSE472-blanket-challenge/486287884d7099c273a570a16aec4d387f60320e/final_project_implementation/submission.csv) |
| 7 | Dhruv, Sahajpreet | [dhruvb26/CSE472-blanket-challenge](https://github.com/dhruvb26/CSE472-blanket-challenge/tree/main) | [`solution/runs/20251124_194116/submission.csv`](https://raw.githubusercontent.com/dhruvb26/CSE472-blanket-challenge/b8c8ca86d3f6d0aa97ba79cd1f6d73ca57018c50/solution/runs/20251124_194116/submission.csv) |
| 8 | Fredo, Anton | [saan-volta/CSE472-blanket-challenge-submission](https://github.com/saan-volta/CSE472-blanket-challenge-submission/tree/main) | [`submission/submission.csv`](https://raw.githubusercontent.com/saan-volta/CSE472-blanket-challenge-submission/942d6b74e42ec51330fdfc8505c992db356819cf/submission/submission.csv) |
| 10 | Ang, Muhammed | [muhammedhunaid/CSE472-blanket-challenge](https://github.com/muhammedhunaid/CSE472-blanket-challenge/tree/muhammed/final-submission) | [`submission.csv`](https://raw.githubusercontent.com/muhammedhunaid/CSE472-blanket-challenge/refs/heads/muhammed/final-submission/submission.csv) |

## Evaluation

```mermaid
flowchart LR
    subgraph Train["**TRAINING**"]
        direction TB
        S1["**Support Set**<br/>D(X, y)"]
        Q1["**Query**<br/>D(X, y)"]
        M1["MB mask: $$ m$$"]
        L1["$$p(y_* \mid x_*, \mathcal{D}_{support}) \quad | \quad p(m \mid \mathcal{D}_{support})$$"]

        S1 --> P1["Model"]
        Q1 --> P1
        P1 --> L1
        M1 --> P1
    end

    subgraph Test["**TESTING**"]
        direction TB
        S2["**Support Set**<br/>(X, y)"]
        Q2["**Query**<br/>(X, ?)"]

        S2 --> P2["Model"]
        Q2 --> P2
        P2 --> O2["**Outputs**<br/>ŷ and m̂"]
    end

    Train ==> Test

    style Train fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style Test fill:#bbdefb,stroke:#1976d2,stroke-width:2px
    style Q1 fill:#ffcdd2,stroke:#d32f2f
    style Q2 fill:#e1bee7,stroke:#7b1fa2
```

We evaluate the model on a set of $N$ unseen tasks $\{\mathcal{T}_j\}_{j=1}^N$ on testing stage.

For each task $\mathcal{T}_j$, the dataset is split into a support set and a query set:

- **Support Set**: $\mathcal{D}^{(j)}_{support} = \{(x_i, y_i)\}_{i=1}^k$
- **Query Set**: $\mathcal{D}^{(j)}_{query} = \{(x_i, y_i)\}_{i=1}^{N_{query}}$

,and the model predicts:

1. The target values $\hat{y}$ for samples in $\mathcal{D}^{(j)}_{query}$.
2. The Markov Blanket mask $\hat{m}^{(j)} \in \{0, 1\}^d$.

### Submission file

On submission dataset, for each dataset (data_id), predict both `y` and `markov_blanket`, and save results in `submission.csv` with the following format:

| data_id | y_pred                    | markov_blanket_pred      |
| ------- | ------------------------- | ------------------------ |
| int     | list of float; shape (n,) | list of int;  shape (d,) |

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
