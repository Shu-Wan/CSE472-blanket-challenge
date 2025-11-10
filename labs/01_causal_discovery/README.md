# Lab 1: Causal Discovery and Markov Blanket

The goal of this lab is to help you get familiar with causal discovery, and
how it can be used for causal feature selection

## Setup

🤗 [Graphs](https://huggingface.co/datasets/CSE472-blanket-challenge/phase1-graphs) | 🤗 [Dataset](https://huggingface.co/datasets/CSE472-blanket-challenge/phase1-dataset)

- Hugging Face Access Token required
- Set up a python environment using uv

## Tasks

1. Get familiar with dataset
   - Read the data card and explore the data
   - Load the dataset, explore and visualize one example
2. Causal discovery based markov blanket search (CD-MB)
   1. Run PC on the dataset
   2. Implement at least one other causal discovery method (e.g., GES, FCI)
   3. Evaluate CD-MB
3. Evaluate and compare CD-MB

## Questions

1. Can PC recover the DAG well? (performance across degree, density, size of mb, training data size etc)
2. Is causal discovery working well on the dataset? If not, what's your hypothesis? (use 3 as a hypothesis)
3. What's the relationship of size of Markov Blanket and density of the graph? (answer 2)
4. Is dedicated causal feature selection method (IAMB) working better than causal discovery methods?

## Bonus (5 points)

1. Implement a specialized causal feature selection method (IAMB)
   - [bnlearn](https://github.com/cran/bnlearn) implements IAMB, fastIAMB. Using [rpy2](https://rpy2.github.io/)
   or write a R script to run it (if you're familiar with R).
   - [py-tetrad](https://github.com/cmu-phil/py-tetrad) also provides IAMB implementations.
2. Is dedicated causal feature selection method (IAMB) working better than causal discovery methods?

## Reading Materials

1. Common causal discovery methods
2. Causal discovery packages
   1. [causal-learn](https://github.com/py-why/causal-learn)
   2. [gcastle](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle)
   3. [pgmpy](https://pgmpy.org/index.html)
   4. [bnlearn, python](https://github.com/erdogant/bnlearn)
   5. [bnlearn, R](https://github.com/cran/bnlearn/tree/master)[^1]
3. Evaluation metrics for causal discovery

[^1]: bnlearn python and R packages are not related.
