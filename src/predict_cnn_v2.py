import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import matplotlib.pyplot as plt


# ============================================================
# 1. 基本配置
# ============================================================

MODEL_PATH = "models/posture_cnn_v2.pt"

DATA_PATH = "data/simulated/posture_dataset_v2.csv"


CLASS_NAMES = [
    "normal",
    "left",
    "right",
    "forward",
    "backward"
]


# ============================================================
# 2. CPU / GPU
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "使用设备：",
    device
)


# ============================================================
# 3. CNN结构
#
# 必须和训练时完全一样
# ============================================================

class PostureCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(
            kernel_size=2
        )

        self.fc1 = nn.Linear(
            32 * 4 * 4,
            64
        )

        self.dropout = nn.Dropout(
            0.20
        )

        self.fc2 = nn.Linear(
            64,
            5
        )


    def forward(self, x):

        x = self.conv1(x)

        x = self.relu(x)

        x = self.conv2(x)

        x = self.relu(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x


# ============================================================
# 4. 创建模型
# ============================================================

model = PostureCNN().to(
    device
)


# ============================================================
# 5. 加载训练好的参数
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


# ============================================================
# 6. 设置为预测模式
# ============================================================

model.eval()


print(
    "CNN模型加载成功！"
)


# ============================================================
# 7. 坐姿预测函数
#
# 输入：
# 8 × 8压力矩阵
#
# 输出：
# 坐姿名称
# 概率分数
# ============================================================

def predict_posture(
    pressure_matrix
):

    pressure_matrix = np.asarray(
        pressure_matrix,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # 检查矩阵大小
    # --------------------------------------------------------

    if pressure_matrix.shape != (
        8,
        8
    ):

        raise ValueError(
            "压力矩阵必须是 8×8"
        )


    # --------------------------------------------------------
    # 限制范围
    # --------------------------------------------------------

    pressure_matrix = np.clip(
        pressure_matrix,
        0,
        100
    )


    # --------------------------------------------------------
    # 与训练时保持一致：
    # 0～100 → 0～1
    # --------------------------------------------------------

    pressure_matrix = (
        pressure_matrix
        /
        100.0
    )


    # --------------------------------------------------------
    # 8×8
    #
    # ↓
    #
    # 1×1×8×8
    # --------------------------------------------------------

    tensor = torch.tensor(
        pressure_matrix,
        dtype=torch.float32
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        device
    )


    # --------------------------------------------------------
    # AI预测
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            tensor
        )


        probabilities = torch.softmax(
            output,
            dim=1
        )


        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()


    predicted_name = CLASS_NAMES[
        predicted_index
    ]


    probability = probabilities[
        0,
        predicted_index
    ].item()


    all_probabilities = (
        probabilities[0]
        .cpu()
        .numpy()
    )


    return (
        predicted_index,
        predicted_name,
        probability,
        all_probabilities
    )


# ============================================================
# 8. 暂时从数据集中抽一个样本测试
#
# 注意：
# 这里只是测试“模型加载+预测”功能，
# 不是重新计算模型准确率。
# ============================================================

df = pd.read_csv(
    DATA_PATH
)


random_index = np.random.randint(
    0,
    len(df)
)


row = df.iloc[
    random_index
]


true_label = int(
    row["label"]
)


pressure = (
    row
    .drop(labels=["label"])
    .values
    .astype(np.float32)
    .reshape(8, 8)
)


# ============================================================
# 9. 调用预测函数
# ============================================================

(
    predicted_index,
    predicted_name,
    probability,
    all_probabilities

) = predict_posture(
    pressure
)


# ============================================================
# 10. 输出结果
# ============================================================

print()
print("=" * 60)

print(
    "样本编号：",
    random_index
)

print(
    "真实坐姿：",
    CLASS_NAMES[true_label]
)

print(
    "AI预测：",
    predicted_name
)

print(
    f"最高概率分数："
    f"{probability * 100:.2f}%"
)

print()

print(
    "所有类别概率："
)


for name, prob in zip(
    CLASS_NAMES,
    all_probabilities
):

    print(
        f"{name:10s}: "
        f"{prob * 100:.2f}%"
    )


print("=" * 60)


# ============================================================
# 11. 显示压力热力图
# ============================================================

plt.imshow(
    pressure,
    cmap="hot",
    vmin=0,
    vmax=100
)

plt.colorbar(
    label="Pressure"
)

plt.title(
    f"True: {CLASS_NAMES[true_label]}"
    f" | "
    f"Predicted: {predicted_name}"
)

plt.xlabel(
    "Sensor X"
)

plt.ylabel(
    "Sensor Y"
)

plt.tight_layout()

plt.show()