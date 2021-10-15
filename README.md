# Network Science Assignment 1

## Overview
We first implement from scratch functions to calculate various graph statistics and measures for our datasets, including the degree distribution, clustering coefficient distribution, Laplacian eigenvalue distribution, degree coefficient and more.

We then implement a Barabasi-Albert (BA) model [1] generator and try to build graphs with the same characteristcs as those studied in the first part. We also experiment with a tweaked BA-model generator that preferentially attaches to nodes with low degree instead of high degree and analzye the graphs it generates.

<br />

![Degree distribution of BA model generated graph with m=30,000 n=2](figs/BA%20model%20n=30000%20m=2/1%20-%20Degree%20Distrib.png)

---
<br />

## Setup

- download the datasets from http://networksciencebook.com/translations/en/resources/data.html
  
- unzip networks.zip into data/
  
- create a virtual environment and activate it 
  ```
  python -m venv .venv 
  .venv/Scripts/activate
  ```
- download the requirements 
    ```
    pip install -r requirements.txt
    ```
- run the code:
    ```
    python ./A1.py
    python ./BA_model.py
    ``` 
---
<br />

## References

[1] Albert R., Barabási A.L. (2002) "Statistical mechanics of complex networks". Reviews of Modern Physics. 74 (1): 47–97. https://arxiv.org/abs/cond-mat/0106096 