# Fraud Detection using GNN-Enhanced XGBoost with Explainable AI

A hybrid fraud detection pipeline on the **Elliptic Bitcoin transaction dataset** (203,769 nodes, 234,355 edges), built on the architecture NVIDIA recommends in its AI Blueprint for fraud detection.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1l8gdBJ81CaCfSWlf01BVbPujBtUbk29s?usp=sharing)

> Click the badge above to run the entire pipeline in your browser — no setup required.

---

## The idea

A single model isn't enough for real-world fraud. This pipeline combines three components, each doing one job:

| Component | Role | Why |
|-----------|------|-----|
| **GCN** (Graph Neural Network) | Feature factory | Reads the transaction graph and learns embeddings that capture network-level risk. Fraud rings are visible in graph structure, not in individual transactions. |
| **XGBoost** | Decision maker | Takes GCN embeddings + original features and makes the final fraud / not-fraud call. Fast, accurate, explainable. |
| **SHAP** | Auditor | Explains *why* each transaction was flagged — required for banking compliance. |

## Architecture

```
Transaction Graph
       │
       ▼
┌──────────────┐     node embeddings      ┌──────────────┐     prediction      ┌──────────┐
│  GCN (2-layer)│ ───────────────────────▶ │   XGBoost    │ ──────────────────▶ │  Fraud?  │
│ message passing│   + original features    │ (+ SMOTE)    │                     └──────────┘
└──────────────┘                           └──────────────┘                          │
                                                                                       ▼
                                                                                ┌──────────┐
                                                                                │   SHAP   │
                                                                                │ why?     │
                                                                                └──────────┘
```

## Pipeline steps

1. Load the Elliptic Bitcoin transaction graph
2. Train a 2-layer GNN — **both GCN and GraphSAGE** — for node classification
3. Extract each GNN's hidden-layer embeddings (network-level risk as numbers)
4. Concatenate embeddings + original tabular features → 229-dim feature vector
5. Balance the rare fraud class with SMOTE (training data only)
6. Train XGBoost on the GNN-enhanced features
7. **Ablation:** compare raw XGBoost vs GCN+XGBoost vs GraphSAGE+XGBoost to prove the GNN adds value
8. **Threshold tuning** via the precision-recall curve (business cost trade-off)
9. **SHAP** explanations for audit-ready compliance
10. **Feature importance** split: original features vs GNN embeddings
11. **Graph visualization** of a fraud node's 2-hop neighbourhood

## Key design decisions

- **Why hybrid?** GNN captures network patterns (mule accounts, fraud rings) that tabular models miss; XGBoost gives speed and explainability that a GNN classifier head lacks.
- **Why the Elliptic dataset?** It's a *real* transaction graph — the academic benchmark for GNN fraud detection — not a tabular dataset forced into a graph.
- **Why SMOTE on training data only?** Balancing the test set leaks information and inflates metrics.
- **Why precision/recall over accuracy?** Fraud is ~10% of labelled data; accuracy is misleading on imbalanced problems.

## Tech stack

`PyTorch Geometric` · `XGBoost` · `SHAP` · `imbalanced-learn (SMOTE)` · `scikit-learn` · `Pandas`

## How to run

**Option 1 — Colab (recommended):** click the badge at the top to open the notebook. Set `Runtime → Change runtime type → GPU`, then `Runtime → Run all`.

**Option 2 — Local:**
```bash
pip install torch-geometric xgboost shap imbalanced-learn scikit-learn pandas matplotlib seaborn
jupyter notebook fraud_detection_gnn.ipynb
```

## What's next

- **FLAG (KDD 2025):** adds LLM-derived text features on top of the graph — deployed in Alipay's production credit-risk system.
- **GraphSAGE:** for scalability to larger graphs in production.
- **Temporal GNNs (TGN):** model how fraud patterns evolve over time, which a static GCN can't.
