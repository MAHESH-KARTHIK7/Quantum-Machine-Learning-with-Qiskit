# ============================================================
# FULL VERSION (CLEAN + FIXED + DECISION BOUNDARY)
# ============================================================

import time, warnings
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)

from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_algorithms.state_fidelities import ComputeUncompute



# ============================================================
# DATASET
# ============================================================
iris = load_iris()
X_full = iris.data
y_full = iris.target

# Binary classification (Setosa vs Versicolor)
idx = y_full < 2
X = X_full[idx][:, 2:4]          # ✅ all 4 features
y = y_full[idx]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale to [0, π]
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print("="*60)
print("TABLE I — Dataset Summary")
print("="*60)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Classes: Setosa(0), Versicolor(1)")
print("="*60)

# ============================================================
# VISUALIZATION
# ============================================================
colors = ['royalblue', 'tomato']
labels = ['Setosa','Versicolor']

fig, axes = plt.subplots(1, 2, figsize=(12,5))

# Raw
for c in [0,1]:
    m = y==c
    axes[0].scatter(X[m,0], X[m,1], c=colors[c], label=labels[c])
axes[0].set_title("Raw Features")
axes[0].legend()

# Scaled
for c in [0,1]:
    axes[1].scatter(X_train_sc[y_train==c,0], X_train_sc[y_train==c,1], c=colors[c], label=f"{labels[c]} Train")
    axes[1].scatter(X_test_sc[y_test==c,0], X_test_sc[y_test==c,1], marker='*', c=colors[c], label=f"{labels[c]} Test")
axes[1].set_title("Scaled Features")
axes[1].legend()

plt.tight_layout()
plt.savefig("fig_01_data.png")
plt.show()

# ============================================================
# CIRCUIT
# ============================================================
feature_map = ZZFeatureMap(feature_dimension=2, reps=2)

print("\nTABLE II — Circuit")
print("Qubits:", feature_map.num_qubits)
print("Depth:", feature_map.decompose().depth())

circuit_fig = feature_map.decompose().draw(output='mpl')
circuit_fig.savefig("fig_02_circuit.png")
plt.show()

# ============================================================
# KERNEL + TRAIN (FIXED)
# ============================================================

sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
# quantum_kernel = FidelityQuantumKernel(
#     fidelity=fidelity,
#     feature_map=feature_map
# )
print("\nTraining QSVC...")
t0 = time.time()

qsvc = QSVC(quantum_kernel=quantum_kernel, C=5.0)
qsvc.fit(X_train_sc, y_train)

train_time = time.time() - t0

# ============================================================
# EVALUATION
# ============================================================
y_pred = qsvc.predict(X_test_sc)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n" + "="*60)
print("TABLE III — Performance")
print("="*60)
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"Training Time: {train_time:.2f}s")
print("="*60)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ============================================================
# CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(ax=ax)
plt.savefig("fig_03_confusion.png")
plt.show()

# ============================================================
# DECISION BOUNDARY (FINAL CLEAN VERSION)
# ============================================================
from scipy.ndimage import zoom

print("\nGenerating decision boundary...")
t_db = time.time()

h = 0.15
x_axis = np.arange(0, np.pi + h, h)
y_axis = np.arange(0, np.pi + h, h)
xx, yy = np.meshgrid(x_axis, y_axis)

grid = np.c_[xx.ravel(), yy.ravel()]
Z = qsvc.predict(grid).reshape(xx.shape).astype(float)

# Smooth
Z = zoom(Z, 4)
xx = zoom(xx, 4)
yy = zoom(yy, 4)

print(f"Computed in {time.time()-t_db:.2f}s")

# Plot
plt.figure(figsize=(7,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
plt.contour(xx, yy, Z, colors='black')

for c in [0,1]:
    plt.scatter(X_train_sc[y_train==c,0], X_train_sc[y_train==c,1], label=f"Train {labels[c]}")
    plt.scatter(X_test_sc[y_test==c,0], X_test_sc[y_test==c,1], marker='*', s=150, label=f"Test {labels[c]}")

plt.title(f"QSVC Decision Boundary (Accuracy: {accuracy*100:.1f}%)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.tight_layout()
plt.savefig("fig_04_decision_boundary.png")
plt.show()

print("\nALL DONE — Outputs saved.")