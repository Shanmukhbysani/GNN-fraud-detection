import torch
import torch.nn.functional as F
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


print("Loading Elliptic Bitcoin Dataset...")
dataset = EllipticBitcoinDataset(root="./data/EllipticBitcoin")
data = dataset[0]

# The Elliptic dataset classes: 0 (licit), 1 (illicit/fraud), 2 (unknown)
# We isolate only the labeled nodes for training and testing
labeled_indices = (data.y != 2).nonzero(as_tuple=False).view(-1)
y = data.y[labeled_indices].cpu().numpy()


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        embeddings = self.conv2(x, edge_index)
        out = self.classifier(F.relu(embeddings))
        return out, embeddings


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(data.num_features, 64, 2).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

print("Training GCN on the transaction graph...")
model.train()

# Basic semi-supervised split for the GCN training phase
indices = labeled_indices.cpu().numpy()
train_idx, _ = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
train_mask[train_idx] = True

# Brief training loop to encode risk patterns into embeddings
for epoch in range(50):
    optimizer.zero_grad()
    out, _ = model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

# Extract the network-level embeddings
model.eval()
with torch.no_grad():
    _, all_embeddings = model(data.x, data.edge_index)

all_embeddings = all_embeddings.cpu().numpy()
original_features = data.x.cpu().numpy()

# Combine tabular features with the new GNN embeddings
hybrid_features = np.hstack((original_features, all_embeddings))

# Filter dataset to only include labeled transactions
X_labeled = hybrid_features[labeled_indices.cpu().numpy()]
y_labeled = y

X_train, X_test, y_train, y_test = train_test_split(
    X_labeled, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled
)

print(f"\nOriginal Training shape: {X_train.shape} | Fraud cases: {sum(y_train)}")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"SMOTE Training shape:    {X_train_sm.shape} | Fraud cases: {sum(y_train_sm)}")

print("\nTraining XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=1,
    random_state=42,
    eval_metric="logloss",
)
xgb_model.fit(X_train_sm, y_train_sm)
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 40)
print("       PERFORMANCE & METRICS AUDIT")
print("=" * 40)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["Licit (0)", "Illicit/Fraud (1)"]))

cm = confusion_matrix(y_test, y_pred)
print("--- Confusion Matrix Breakdown ---")
print(f"True Negatives  (Licit cleared)     : {cm[0][0]}")
print(f"False Positives (Licit flagged)     : {cm[0][1]}  <-- Customer friction")
print(f"False Negatives (Fraud missed)      : {cm[1][0]}  <-- Financial loss")
print(f"True Positives  (Fraud caught)      : {cm[1][1]}  <-- Successful catches")

roc_auc = roc_auc_score(y_test, y_proba)
print(f"\nROC-AUC Score : {roc_auc:.4f}")

precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)
print(f"PR-AUC Score  : {pr_auc:.4f}")


def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=["Predicted Licit", "Predicted Fraud"],
        yticklabels=["Actual Licit", "Actual Fraud"],
    )
    plt.title("Hybrid XGBoost Fraud Detection")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


# Uncomment the line below to view the plot
# plot_confusion_matrix(cm)

print("\nGenerating SHAP feature attributions for flagged transactions...")
explainer = shap.TreeExplainer(xgb_model)

flagged_indices = np.where(y_pred == 1)[0]

if len(flagged_indices) > 0:
    audit_sample = X_test[flagged_indices[:50]]
    shap_values = explainer.shap_values(audit_sample)

    print(f"Successfully generated SHAP values for {len(audit_sample)} flagged transactions.")
    # shap.summary_plot(shap_values, audit_sample, plot_type="bar")
else:
    print("No fraud flagged in this subset.")