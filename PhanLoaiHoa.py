# -*- coding: utf-8 -*-
"""
Iris + Pipeline(StandardScaler -> GaussianNB) + CV:
- Stratified train/test split
- GridSearchCV (StratifiedKFold) tối ưu var_smoothing
- In best params, best CV score, test metrics
- Vẽ & lưu Confusion Matrix (matplotlib)
- Dự đoán mẫu mới + vẽ & lưu biểu đồ xác suất dự đoán
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
)

# ---------- Helpers (matplotlib only) ----------
def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], acc: float, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    im = ax.imshow(cm, interpolation="nearest")  # default colormap
    ax.set_title(f"Confusion Matrix (Accuracy={acc:.4f})")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(class_names)

    # annotate từng ô
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_probabilities(proba: np.ndarray, class_names: list[str], predicted_label: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    bars = ax.bar(class_names, proba)  # không set màu thủ công
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Probability")
    ax.set_title(f"Predicted probabilities for new sample → {predicted_label}")

    for rect, val in zip(bars, proba):
        ax.text(rect.get_x() + rect.get_width() / 2.0, val + 0.02, f"{val:.2f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ---------- Main ----------
def main() -> None:
    # Đường lưu hình
    cm_path = Path("confusion_matrix.png")
    prob_path = Path("predicted_probabilities.png")

    # 1) Dữ liệu
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names.tolist()

    # 2) Chia tập (giữ tỉ lệ lớp)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Pipeline: Chuẩn hoá -> GaussianNB
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ]
    )
    
# 4) Thiết lập StratifiedKFold & GridSearchCV
    # var_smoothing thường quét theo logspace
    param_grid = {
        "clf__var_smoothing": np.logspace(-12, -6, 13)  # 1e-12 ... 1e-6
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="accuracy",
        cv=skf,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )

    # 5) Huấn luyện với GridSearchCV trên tập train
    grid.fit(X_train, y_train)

    print("===> GridSearchCV ket quan tot nhat")
    print("Best params:", grid.best_params_)
    print(f"Best CV accuracy (mean over folds): {grid.best_score_:.4f}")

    # 6) Đánh giá trên tập test
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)

    print("\n===> Danh gia tren tep TEST")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix (rows=thuc, cols=du doan):\n", cm)
    print("Classification Report:\n", report)

    # 7) Vẽ/Lưu Confusion Matrix
    plot_confusion_matrix(cm, class_names, acc, cm_path)
    print(f"Da luu hinh Confusion Matrix → {cm_path.resolve()}")

    # 8) Dự đoán mẫu mới + xác suất
    new_sample = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=float)
    proba = best_model.predict_proba(new_sample)[0]
    pred_label = class_names[int(np.argmax(proba))]

    print("Mau moi:", new_sample.tolist())
    print("Loai hoa du doan:", pred_label)
    print("Xac xuat du doan theo lop:", dict(zip(class_names, map(float, proba))))

    plot_probabilities(proba, class_names, pred_label, prob_path)
    print(f"Da luu hinh xac suat du doan → {prob_path.resolve()}")


if __name__ == "__main__":
    main()