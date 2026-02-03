# MetaFAM Explorer

MetaFAM Explorer is a Streamlit-based dashboard for structural analysis and interactive exploration of large-scale kinship and relationship graphs constructed from triplet data.

The tool is for analytical inspection and debugging of inferred family networks rather than end-user visualization or presentation.

---

## Functionality

- Ingests directed triplet data of the form *(head, relation, tail)*
- Constructs a directed multigraph over individuals
- Computes graph-level and node-level statistics
- Supports interactive exploration through multiple analytical views
- Identifies founders, leaves, bridges, and anomalous structures

---

## Input Data

Input is a delimited text file where each line represents a directed relation:

Relation semantics are defined in `data_backend.py`.

---

## Usage

Run the dashboard from the project root:

```bash
streamlit run app.py
```

The default expected data path is:

```text
../data/train.txt
```

This can be modified from the sidebar.

---

## Views

- **Overview**: global graph statistics and distributions
- **Graph Explorer**: family tree, ego network, and sampled subgraph views
- **Path Finder**: enumeration and visualization of relational paths between two nodes
- **Person Lookup**: detailed node-level attributes and relations
- **Statistics**: degree distributions, high-degree nodes, anomaly summaries, and exports

---

## Dependencies

Core dependencies include:

- streamlit
- pandas
- plotly
- networkx
- scipy (required for centrality computations)

---

## Status

Research and analysis tool. Not optimized for production deployment.