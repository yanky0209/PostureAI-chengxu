import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. 读取 V2 数据
# ============================================================

data_path = "data/simulated/posture_dataset_v2.csv"

df = pd.read_csv(data_path)

print("=" * 60)
print("V2 数据读取成功！")
print("数据形状：", df.shape)
print("=" * 60)


# ============================================================
# 2. 特征 X 和标签 y
# ============================================================

X = df.drop(columns=["label"])
y = df["label"]

print("样本数量：", X.shape[0])
print("特征数量：", X.shape[1])
print("类别：", sorted(y.unique()))


# ============================================================
# 3. 划分训练集和测试集
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print()
print("训练集：", X_train.shape)
print("测试集：", X_test.shape)


# ============================================================
# 4. 标准化
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 5. 创建 SVM
# ============================================================

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)


# ============================================================
# 6. 训练
# ============================================================

print()
print("=" * 60)
print("开始训练 V2 SVM...")
print("=" * 60)

model.fit(
    X_train_scaled,
    y_train
)

print("训练完成！")


# ============================================================
# 7. 预测
# ============================================================

y_pred = model.predict(
    X_test_scaled
)


# ============================================================
# 8. 准确率
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print()
print("=" * 60)
print("SVM V2 测试结果")
print("=" * 60)

print(
    f"准确率：{accuracy:.4f}"
)

print(
    f"准确率百分比：{accuracy * 100:.2f}%"
)


# ============================================================
# 9. 分类报告
# ============================================================

class_names = [
    "normal",
    "left",
    "right",
    "forward",
    "backward"
]

print()
print("详细分类结果：")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=class_names
    )
)


# ============================================================
# 10. 混淆矩阵
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3, 4]
)

print("混淆矩阵：")
print(cm)


# ============================================================
# 11. 绘图
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(cm)

plt.title(
    "SVM V2 Posture Classification"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    range(5),
    class_names,
    rotation=30
)

plt.yticks(
    range(5),
    class_names
)

for i in range(5):

    for j in range(5):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.show()