import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. 固定随机种子
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 2. 基本配置
# ============================================================

DATA_PATH = "data/simulated/posture_dataset_v2.csv"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "posture_cnn_v2.pt"
)

BATCH_SIZE = 64

EPOCHS = 30

LEARNING_RATE = 0.001


CLASS_NAMES = [
    "normal",
    "left",
    "right",
    "forward",
    "backward"
]


# ============================================================
# 3. 选择 CPU / GPU
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("使用设备：", device)
print("=" * 60)


# ============================================================
# 4. 读取数据
# ============================================================

df = pd.read_csv(
    DATA_PATH
)

print()
print("数据读取成功！")

print(
    "数据形状：",
    df.shape
)


# ============================================================
# 5. 提取 X 和 y
# ============================================================

X = df.drop(
    columns=["label"]
).values.astype(
    np.float32
)

y = df[
    "label"
].values.astype(
    np.int64
)


# ============================================================
# 6. 把压力值归一化到 0～1
# ============================================================

X = X / 100.0


# ============================================================
# 7. 64维重新变回 8×8
#
# CNN需要：
#
# 样本数 × 通道数 × 高 × 宽
#
# N × 1 × 8 × 8
# ============================================================

X = X.reshape(
    -1,
    1,
    8,
    8
)

print(
    "CNN输入形状：",
    X.shape
)


# ============================================================
# 8. 第一次划分
#
# 80%：暂时作为训练+验证
# 20%：最终测试集
# ============================================================

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)


# ============================================================
# 9. 第二次划分
#
# 从剩余80%中再拿20%做验证集
#
# 最终：
#
# 64% train
# 16% validation
# 20% test
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.20,
    random_state=SEED,
    stratify=y_train_val
)


print()

print(
    "训练集：",
    X_train.shape
)

print(
    "验证集：",
    X_val.shape
)

print(
    "测试集：",
    X_test.shape
)


# ============================================================
# 10. NumPy → PyTorch Tensor
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)


X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val,
    dtype=torch.long
)


X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)


# ============================================================
# 11. 创建 Dataset
# ============================================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)


# ============================================================
# 12. DataLoader
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 13. 定义轻量 CNN
# ============================================================

class PostureCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # 第一层卷积
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        # 第二层卷积
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # ReLU 激活
        self.relu = nn.ReLU()

        # 池化
        self.pool = nn.MaxPool2d(
            kernel_size=2
        )

        # 全连接层
        self.fc1 = nn.Linear(
            32 * 4 * 4,
            64
        )

        # Dropout
        self.dropout = nn.Dropout(
            0.20
        )

        # 最终5分类
        self.fc2 = nn.Linear(
            64,
            5
        )


    def forward(self, x):

        # 输入：
        # N × 1 × 8 × 8

        x = self.conv1(x)

        x = self.relu(x)

        # N × 16 × 8 × 8


        x = self.conv2(x)

        x = self.relu(x)

        # N × 32 × 8 × 8


        x = self.pool(x)

        # N × 32 × 4 × 4


        x = torch.flatten(
            x,
            start_dim=1
        )

        # N × 512


        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)


        x = self.fc2(x)

        # N × 5

        return x


# ============================================================
# 14. 创建模型
# ============================================================

model = PostureCNN().to(
    device
)

print()
print(model)


# ============================================================
# 15. 损失函数
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 16. 优化器
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 17. 用于保存训练历史
# ============================================================

train_losses = []

val_losses = []

train_accuracies = []

val_accuracies = []


best_val_accuracy = 0.0


# ============================================================
# 18. 正式训练
# ============================================================

print()
print("=" * 60)
print("开始训练 CNN")
print("=" * 60)


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # 训练模式
    # --------------------------------------------------------

    model.train()

    total_train_loss = 0.0

    train_correct = 0

    train_total = 0


    for inputs, labels in train_loader:

        inputs = inputs.to(
            device
        )

        labels = labels.to(
            device
        )


        # 清空旧梯度
        optimizer.zero_grad()


        # 前向传播
        outputs = model(
            inputs
        )


        # 计算损失
        loss = criterion(
            outputs,
            labels
        )


        # 反向传播
        loss.backward()


        # 更新参数
        optimizer.step()


        total_train_loss += (
            loss.item()
            *
            inputs.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        train_total += labels.size(0)

        train_correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        total_train_loss
        /
        train_total
    )


    train_accuracy = (
        train_correct
        /
        train_total
    )


    # --------------------------------------------------------
    # 验证模式
    # --------------------------------------------------------

    model.eval()

    total_val_loss = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for inputs, labels in val_loader:

            inputs = inputs.to(
                device
            )

            labels = labels.to(
                device
            )


            outputs = model(
                inputs
            )


            loss = criterion(
                outputs,
                labels
            )


            total_val_loss += (
                loss.item()
                *
                inputs.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        total_val_loss
        /
        val_total
    )


    val_accuracy = (
        val_correct
        /
        val_total
    )


    # --------------------------------------------------------
    # 保存历史
    # --------------------------------------------------------

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    val_accuracies.append(
        val_accuracy
    )


    # --------------------------------------------------------
    # 保存最好的模型
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )

        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


    print(
        f"Epoch "
        f"{epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}%"
    )


# ============================================================
# 19. 加载验证集表现最好的模型
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


print()
print("=" * 60)

print(
    f"最佳验证集准确率："
    f"{best_val_accuracy * 100:.2f}%"
)

print("=" * 60)


# ============================================================
# 20. 测试集预测
# ============================================================

all_predictions = []

all_labels = []


with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(
            device
        )

        outputs = model(
            inputs
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# ============================================================
# 21. 测试准确率
# ============================================================

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)


print()
print("=" * 60)
print("CNN V2 最终测试结果")
print("=" * 60)

print(
    f"测试准确率："
    f"{test_accuracy:.4f}"
)

print(
    f"测试准确率百分比："
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# 22. 分类报告
# ============================================================

print()
print("详细分类结果：")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=[
            0,
            1,
            2,
            3,
            4
        ],
        target_names=CLASS_NAMES
    )
)


# ============================================================
# 23. 混淆矩阵
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[
        0,
        1,
        2,
        3,
        4
    ]
)


print(
    "混淆矩阵："
)

print(cm)


# ============================================================
# 24. 绘制混淆矩阵
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(cm)

plt.title(
    "CNN V2 Posture Classification"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)


plt.xticks(
    range(5),
    CLASS_NAMES,
    rotation=30
)

plt.yticks(
    range(5),
    CLASS_NAMES
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


# ============================================================
# 25. 训练 / 验证 Loss 曲线
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    train_losses,
    label="Train Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "CNN Training Loss"
)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 26. 训练 / 验证 Accuracy 曲线
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    train_accuracies,
    label="Train Accuracy"
)

plt.plot(
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "CNN Training Accuracy"
)

plt.legend()

plt.tight_layout()

plt.show()


print()
print("=" * 60)

print(
    "CNN 模型已保存：",
    MODEL_PATH
)

print("=" * 60)