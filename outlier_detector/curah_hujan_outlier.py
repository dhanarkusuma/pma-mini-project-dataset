import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("datasets/train.csv")

Q1 = df["curah_hujan"].quantile(q=0.25)
Q3 = df["curah_hujan"].quantile(q=0.75)
IQR: float = Q3 - Q1

lower_bound: float = Q1 - 1.5 * IQR
upper_bound: float = Q3 + 1.5 * IQR

outlier = (df["curah_hujan"] < lower_bound) | (df["curah_hujan"] > upper_bound)

y_jitter = np.random.normal(0, 0.02, size=len(df["curah_hujan"]))

plt.figure(figsize=(9, 3))
plt.scatter(df["curah_hujan"][~outlier], y_jitter[~outlier], label="Normal", alpha=0.7)
plt.scatter(df["curah_hujan"][outlier], y_jitter[outlier], label="Outlier", alpha=0.9)

plt.axvline(float(Q1), linestyle="--", linewidth=1, label="Q1")
plt.axvline(float(Q3), linestyle="--", linewidth=1, label="Q3")
plt.axvline(lower_bound, linestyle=":", linewidth=2, label="Lower Bound")
plt.axvline(upper_bound, linestyle=":", linewidth=2, label="Upper Bound")

plt.xlabel("Curah Hujan")
plt.yticks([])
plt.title("Univariate Scatterplot Curah Hujan (IQR)")
plt.legend()
plt.tight_layout()
plt.show()


print(f"Q1: {Q1:.2f}")
print(f"Q3: {Q3:.2f}")
print(f"IQR: {IQR:.2f}")
print(f"Lower Bound (Batas Bawah): {lower_bound:.2f}")
print(f"Upper Bound (Batas Atas): {upper_bound:.2f}")

jumlah_outlier = outlier.sum()
print(f"Jumlah outlier yang terdeteksi: {jumlah_outlier}")

