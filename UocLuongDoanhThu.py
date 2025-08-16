# -*- coding: utf-8 -*-
"""
Linear Regression dự đoán doanh thu/ngày
- Đọc CSV an toàn (absolute path / cùng thư mục script)
- Pipeline: One-Hot + StandardScaler + LinearRegression
- Hold-out metrics + KFold CV (R^2 mean ± std)
- Plot: y_thật vs y_dự đoán, Residuals vs ŷ
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor


from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# 0) Xác định đường dẫn CSV
# ----------------------------
# Cách 1: đặt đường dẫn tuyệt đối vào đây nếu bạn muốn cố định vị trí file
csv_path_abs = "D:/Users/Learn/CaoHoc/HocMay/workspace/Data/Mall_Customers.csv"  # ví dụ: "/Users/thanhha/Desktop/du_lieu_cua_hang_ban_le.csv"

def load_csv_safe(csv_path_abs: str) -> pd.DataFrame:
    # 1) Nếu có đường dẫn tuyệt đối hợp lệ -> dùng luôn
    if csv_path_abs:
        p = Path(csv_path_abs).expanduser()
        if p.exists():
            return pd.read_csv(p, parse_dates=["CustomerID"])
        else:
            print(f"[Cảnh báo] Không tìm thấy CSV tại absolute path: {p}", file=sys.stderr)

    # 2) Thử cùng thư mục với file .py
    try:
        here = Path(__file__).parent
    except NameError:
        # khi chạy trong notebook hoặc interactive, __file__ không có
        here = Path.cwd()
    p2 = here / "du_lieu_cua_hang_ban_le.csv"
    if p2.exists():
        return pd.read_csv(p2, parse_dates=["date"])

    # 3) Thử thư mục làm việc hiện tại
    p3 = Path.cwd() / "du_lieu_cua_hang_ban_le.csv"
    if p3.exists():
        return pd.read_csv(p3, parse_dates=["date"])

    raise FileNotFoundError(
        "Không tìm thấy 'du_lieu_cua_hang_ban_le.csv'. "
        "Hãy tải file CSV và đặt cạnh script, hoặc điền csv_path_abs bằng đường dẫn tuyệt đối."
    )

# ----------------------------
# 1) Đọc dữ liệu
# ----------------------------
df = load_csv_safe(csv_path_abs)

print("5 dòng đầu:")
print(df.head(), "\n")
print("Mô tả tổng quan:")
print(df.describe(include="all"))

# ----------------------------
# 2) EDA nhanh & tiền xử lý tối thiểu
# # ----------------------------
print("\nThiếu dữ liệu theo cột:")
print(df.isna().sum())

target = "Spending Score (1-100)"
num_features = ["Age", "Annual Income (k$)"]
# xử lý như categorical (string): promo_level, ngay_trong_tuan, thang, su_kien_dac_biet
cat_features = ["Gender"]

X = df[num_features + cat_features]
y = df[target]

# # ----------------------------
# # 3) Tách tập train/test
# # ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# # ----------------------------
# # 4) Pipeline: One-hot + Scale + LinearRegression
# # ----------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
    ],
    remainder="drop",
)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("lr", LinearRegression()),
    ]
)

# # ----------------------------
# # 5) Huấn luyện & đánh giá Hold-out
# # ----------------------------
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = float(np.sqrt(mse))
r2   = float(r2_score(y_test, y_pred))

print("\nHieu nang tren tap TEST (hold-out):")
print({"MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 3)})

# # ----------------------------
# # 6) Trực quan
# # ----------------------------
plt.figure(figsize=(6, 5), dpi=120)
plt.scatter(y_test, y_pred, alpha=0.7)
y_min = float(min(y_test.min(), y_pred.min()))
y_max = float(max(y_test.max(), y_pred.max()))
plt.plot([y_min, y_max], [y_min, y_max], color="red", linestyle="--")
plt.xlabel("Giá trị thực (Spending Score)")
plt.ylabel("Giá trị dự đoán")
plt.title("So sánh y_thật vs y_dự đoán")
plt.tight_layout()
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(6, 5), dpi=120)
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Giá trị dự đoán")
plt.ylabel("Phần dư (y - ŷ)")
plt.title("Residuals vs Predicted")
plt.tight_layout()
plt.show()

# # ----------------------------
# # 7) Cross-Validation (KFold) với R²
# # ----------------------------
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")

print(f"\nKFold CV ({k} folds) - R² từng fold:", np.round(cv_scores, 3).tolist())
print(f"R² trung bình (CV): {cv_scores.mean():.3f}")
print(f"Độ lệch chuẩn R² (CV): {cv_scores.std():.3f}")

# ----------------------------
# 8) Gợi ý mở rộng
# ----------------------------
# - Thử Ridge/Lasso để regularize và giảm đa cộng tuyến
# - Dùng TimeSeriesSplit nếu dữ liệu có thứ tự thời gian dài
# - Thêm đặc trưng (cuối tuần, lễ, thời tiết, lag features)

# Mô hình phi tuyến: Random Forest
rf_pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),  # Dùng chung bộ preprocess
        ("rf", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )),
    ]
)

# # Huấn luyện và đánh giá trên hold-out
# rf_pipe.fit(X_train, y_train)
# y_pred_rf = rf_pipe.predict(X_test)

# mae_rf  = mean_absolute_error(y_test, y_pred_rf)
# mse_rf  = mean_squared_error(y_test, y_pred_rf)
# rmse_rf = float(np.sqrt(mse_rf))
# r2_rf   = float(r2_score(y_test, y_pred_rf))

# print("\nHiệu năng trên tập TEST (hold-out) - Random Forest:")
# print({"MAE": round(mae_rf, 3), "RMSE": round(rmse_rf, 3), "R2": round(r2_rf, 3)})

# # Cross-Validation cho Random Forest
# cv_scores_rf = cross_val_score(rf_pipe, X, y, cv=kf, scoring="r2")
# print(f"\nKFold CV ({k} folds) - R² từng fold (Random Forest):", np.round(cv_scores_rf, 3).tolist())
# print(f"R² trung bình (CV - RF): {cv_scores_rf.mean():.3f}")
# print(f"Độ lệch chuẩn R² (CV - RF): {cv_scores_rf.std():.3f}")

# # ----------------------------
# # 10) So sánh kết quả
# # ----------------------------
# print("\n=== So sánh Tuyến tính vs Phi tuyến (CV Mean R²) ===")
# print(f"Linear Regression: {cv_scores.mean():.3f}")
# print(f"Random Forest:     {cv_scores_rf.mean():.3f}")