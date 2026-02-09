# Report 

Check Graphs_Report.pdf uploaded above. Or use the link [HERE](https://drive.google.com/file/d/1Fbr6IW7MSzhkcGsu_JHn41KrI2OGJo3N/view?usp=sharing)

# How to Run

## Setup

I decided to use two virtual environments in the project root (and I would recommend you do the same):

- **`venv`** — for Tasks 1–3 (install from `requirements.txt`)
- **`venv2`** — for Task 4 notebooks (PyTorch + standard ML stack)

`venv2` is kept separate because PyTorch is a large dependency (~8GB) and isn't needed for the rest of the project.

```bash
# Tasks 1-3
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Task 4
python -m venv venv2
source venv2/bin/activate
pip install numpy matplotlib scikit-learn torch
```

**IMPORTANT** All scripts are run from the **project root**. The module convention is:

```bash
python -m src.taskN.filename
```

---


### Dashboard

```bash
streamlit run dashboard/app.py
```
Interactive Streamlit + PyVis dashboard for visually exploring family graphs, filtering by relation type, and inspecting individual families.

---

## Task 1 — Data Exploration & Feature Extraction

```bash
python -m src.task1.primary_analysis
```
Loads the full dataset, extracts per-person features (degree, relation counts, gender inference), and saves structured CSVs for downstream use.

```bash
python -m src.task1.secondary_analysis
```
Computes standard graph-theoretic metrics (centrality, clustering coefficients, degree distributions) plus family-specific statistics.

```bash
python -m src.task1.tertiary_analysis
```
Custom domain-aware centrality metrics designed specifically for family knowledge graphs (generational depth, bridging roles, etc.).


---

## Task 2 — Community Detection & Graph Analysis

```bash
python -m src.task2.autorunner
```
Runs all community detection techniques (Louvain, spectral clustering, Node2Vec + KMeans) on both the full graph and nuclear-family subgraphs, and prints comparative results. Shows how standard algorithms fail on dense kinship graphs but succeed on nuclear family projections.

```bash
python -m src.task2.cross_family_isomorphism
```
Tests structural similarity across the 50 families using graph isomorphism — confirms all families are structurally distinct.

```bash
python -m src.task2.ancestor_overlap
```
Computes pairwise ancestor Jaccard similarity — siblings and same-parent pairs score ~1.0, cousins score lower.

```bash
python -m src.task2.relation_weighed_distance
```
Weighted shortest-path distances using relation-type-aware edge weights (parent edges weigh differently than sibling edges).

```bash
python -m src.task2.fragility_v_redundancy
```
Node removal analysis — removes each person from a family and measures what breaks (components, diameter, lost edges).

```bash
python -m src.task2.spearman_correlation
```
Validates that weighted distance and ancestor overlap are correlated via Spearman rank correlation.

```bash
python -m src.task2.family_disjoint_verification
```
Manual BFS verification that the 50 families are truly disjoint connected components.

---

## Task 3 — Rule Mining

```bash
python -m src.task3.task3_main
```
Orchestrates all four rule mining steps in sequence (can also be run individually below).

```bash
python -m src.task3.rule_miner_amie
```
AMIE-style inductive rule mining — discovers two-hop compositional rules (e.g., `motherOf ∧ sisterOf → auntOf`) with minimum support filtering.

```bash
python -m src.task3.ik_these_relations_already
```
Domain-validated rules — evaluates known common-sense family rules (inverse pairs, symmetry, compositions) against the dataset.

```bash
python -m src.task3.pca_confidence
```
PCA (Partial Completeness Assumption) confidence — a more appropriate confidence metric for open-world KGs where missing ≠ false.

---

## Task 4 — Link Prediction

> **Uses `venv2`.** Link the `venv2` kernel to Jupyter to run these notebooks.

The four notebooks form a progression from simple heuristics to graph neural networks:

| Notebook | Description |
|---|---|
| `0_baselines.ipynb` | Heuristic baselines. Random, popularity-based, and rule-based link prediction |
| `1_distmult.ipynb` | DistMult knowledge graph embedding model as a lightweight learned baseline |
| `2_rgcn_v5.ipynb` | R-GCN for relation-aware link prediction |
| `3_analysis.ipynb` | Comparative analysis, per-relation breakdowns, and embedding visualizations |

`task4_utils.py` contains shared data loading, negative sampling, and evaluation utilities used by all notebooks.