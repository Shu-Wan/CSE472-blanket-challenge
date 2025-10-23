# 01 Causal Discovery and Markov Blanket

The goal of this lab is to help you get familiar with causal discovery, and
how it can be used for causal feature selection

## Setup

🤗 Dataset: https://huggingface.co/datasets/CSE472-blanket-challenge/phase1-dataset

- Hugging Face Access Token required
- Set up a python environment using uv

## Tasks

1. Get familiar with dataset
   - Read the data card and explore the data
   - Load the dataset, explore and visualize one example
2. Causal discovery
   1. Run PC on the dataset
   2. Implement at least one other causal discovery method (e.g., GES, FCI)
   3. Evaluate causal discovery methods
3. Find the Markov Blanket
   1. CD-based MB extraction
   2. Implement specialized causal feature selection methods (IAMB)
4. Try answering the questions below

## Questions

1. Can PC recover the DAG well?
2. Is causal discovery working well on the dataset? If not, what's your hypothesis?
3. What's the relationship of size of Markov Blanket and density of the graph?
4. Is dedicated causal feature selection method (IAMB) working better than causal discovery methods?

## Reading Materials

1. Common causal discovery methods (pc and other family of CF)
2. Causal discovery packages
   1. [causal-learn](https://github.com/py-why/causal-learn)
   2. [gcastle](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle)
   3. [pgmpy](https://pgmpy.org/index.html)
   4. [bnlearn, python*](https://github.com/erdogant/bnlearn)
   5. [bnlearn, R*](https://github.com/cran/bnlearn/tree/master)
3. Evaluation metrics for causal discovery

---
*: bnlearn python and R packages are not related.
