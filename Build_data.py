import numpy as np
import pandas as pd

np.random.seed(42)
n = 200

# Giả lập dữ liệu
CustomerID = np.arange(1, n+1)
Gender = np.random.choice(["Male", "Female"], size=n)
Age = np.random.randint(18, 65, size=n)
Income = np.random.randint(15, 140, size=n)

# Tạo Spending Score tuyến tính: w1*Age + w2*Income + noise
y_true = 0.4 * Age + 0.6 * Income + np.random.normal(0, 5, size=n)
y_scaled = np.clip(y_true, 1, 100)

# Làm tròn và chuyển thành int
y_scaled = np.round(y_scaled).astype(int)

df_linear = pd.DataFrame({
    "CustomerID": CustomerID,
    "Gender": Gender,
    "Age": Age,
    "Annual Income (k$)": Income,
    "Spending Score (1-100)": y_scaled
})

print(df_linear.head())
df_linear.to_csv("mall_customers_linear.csv", index=False)
