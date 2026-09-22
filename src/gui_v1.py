import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout
)

from PyQt6.QtCore import Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


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


CHINESE_NAMES = {
    "normal": "正常坐姿",
    "left": "左倾",
    "right": "右倾",
    "forward": "前倾",
    "backward": "后仰"
}


# ============================================================
# 2. CPU / GPU
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 3. CNN模型结构
#
# 必须与训练时一致
# ============================================================

class PostureCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(
            1,
            16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(
            32 * 4 * 4,
            64
        )

        self.dropout = nn.Dropout(0.20)

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
# 4. 加载 CNN
# ============================================================

model = PostureCNN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ============================================================
# 5. 加载模拟数据集
# ============================================================

df = pd.read_csv(
    DATA_PATH
)


# ============================================================
# 6. AI预测函数
# ============================================================

def predict_posture(pressure_matrix):

    pressure_matrix = np.asarray(
        pressure_matrix,
        dtype=np.float32
    )

    pressure_matrix = np.clip(
        pressure_matrix,
        0,
        100
    )

    # 与CNN训练保持一致
    normalized = pressure_matrix / 100.0

    tensor = torch.tensor(
        normalized,
        dtype=torch.float32
    )

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)


    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        index = torch.argmax(
            probabilities,
            dim=1
        ).item()


    name = CLASS_NAMES[index]

    probability = probabilities[
        0,
        index
    ].item()


    return (
        index,
        name,
        probability
    )


# ============================================================
# 7. Matplotlib热力图组件
# ============================================================

class PressureCanvas(FigureCanvasQTAgg):

    def __init__(self):

        self.figure = Figure(
            figsize=(6, 6)
        )

        super().__init__(
            self.figure
        )

        self.ax = self.figure.add_subplot(111)


        empty_data = np.zeros(
            (8, 8)
        )


        self.image = self.ax.imshow(
            empty_data,
            cmap="hot",
            vmin=0,
            vmax=100
        )


        self.figure.colorbar(
            self.image,
            ax=self.ax,
            label="Pressure"
        )


        self.ax.set_title(
            "Pressure Distribution"
        )

        self.ax.set_xlabel(
            "Sensor X"
        )

        self.ax.set_ylabel(
            "Sensor Y"
        )


        self.figure.tight_layout()


    def update_pressure(
        self,
        pressure
    ):

        self.image.set_data(
            pressure
        )

        self.draw_idle()


# ============================================================
# 8. GUI主窗口
# ============================================================

class PostureWindow(QMainWindow):

    def __init__(self):

        super().__init__()


        # ----------------------------------------------------
        # 窗口基本设置
        # ----------------------------------------------------

        self.setWindowTitle(
            "PostureAI 智能坐姿监测系统"
        )

        self.resize(
            1000,
            650
        )


        # ----------------------------------------------------
        # 主控件
        # ----------------------------------------------------

        central_widget = QWidget()

        self.setCentralWidget(
            central_widget
        )


        main_layout = QVBoxLayout(
            central_widget
        )


        # ====================================================
        # 标题
        # ====================================================

        title = QLabel(
            "PostureAI 智能坐姿监测系统"
        )

        title.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        title.setStyleSheet(
            """
            font-size: 28px;
            font-weight: bold;
            padding: 15px;
            """
        )


        main_layout.addWidget(
            title
        )


        # ====================================================
        # 中间区域
        # ====================================================

        content_layout = QHBoxLayout()


        # ----------------------------------------------------
        # 左侧：压力热力图
        # ----------------------------------------------------

        self.canvas = PressureCanvas()

        content_layout.addWidget(
            self.canvas,
            2
        )


        # ----------------------------------------------------
        # 右侧信息区
        # ----------------------------------------------------

        info_layout = QVBoxLayout()


        current_title = QLabel(
            "当前坐姿"
        )

        current_title.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        current_title.setStyleSheet(
            """
            font-size: 20px;
            font-weight: bold;
            """
        )


        self.posture_label = QLabel(
            "等待检测"
        )

        self.posture_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.posture_label.setStyleSheet(
            """
            font-size: 36px;
            font-weight: bold;
            padding: 20px;
            """
        )


        self.probability_label = QLabel(
            "模型概率：--"
        )

        self.probability_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.probability_label.setStyleSheet(
            """
            font-size: 20px;
            padding: 10px;
            """
        )


        self.status_label = QLabel(
            "系统等待检测"
        )

        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.status_label.setStyleSheet(
            """
            font-size: 22px;
            font-weight: bold;
            padding: 20px;
            """
        )


        info_layout.addStretch()

        info_layout.addWidget(
            current_title
        )

        info_layout.addWidget(
            self.posture_label
        )

        info_layout.addWidget(
            self.probability_label
        )

        info_layout.addWidget(
            self.status_label
        )

        info_layout.addStretch()


        content_layout.addLayout(
            info_layout,
            1
        )


        main_layout.addLayout(
            content_layout
        )


        # ====================================================
        # 按钮
        # ====================================================

        self.simulate_button = QPushButton(
            "模拟一次"
        )

        self.simulate_button.setMinimumHeight(
            55
        )

        self.simulate_button.setStyleSheet(
            """
            font-size: 20px;
            font-weight: bold;
            """
        )


        self.simulate_button.clicked.connect(
            self.simulate_once
        )


        main_layout.addWidget(
            self.simulate_button
        )


    # ========================================================
    # 9. 模拟一次压力数据
    # ========================================================

    def simulate_once(self):

        # ----------------------------------------------------
        # 随机选一条数据
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # AI预测
        # ----------------------------------------------------

        (
            predicted_index,
            predicted_name,
            probability

        ) = predict_posture(
            pressure
        )


        # ----------------------------------------------------
        # 更新热力图
        # ----------------------------------------------------

        self.canvas.update_pressure(
            pressure
        )


        # ----------------------------------------------------
        # 更新坐姿文字
        # ----------------------------------------------------

        chinese_name = CHINESE_NAMES[
            predicted_name
        ]


        self.posture_label.setText(
            chinese_name
        )


        self.probability_label.setText(
            f"模型概率："
            f"{probability * 100:.2f}%"
        )


        # ----------------------------------------------------
        # 判断是否为不良坐姿
        # ----------------------------------------------------

        if predicted_name == "normal":

            self.status_label.setText(
                "✓ 坐姿正常"
            )

            self.status_label.setStyleSheet(
                """
                font-size: 22px;
                font-weight: bold;
                padding: 20px;
                color: green;
                """
            )


        else:

            self.status_label.setText(
                "⚠ 请调整坐姿"
            )

            self.status_label.setStyleSheet(
                """
                font-size: 22px;
                font-weight: bold;
                padding: 20px;
                color: red;
                """
            )


        # ----------------------------------------------------
        # 控制台也打印
        # ----------------------------------------------------

        print()
        print("=" * 50)

        print(
            "真实坐姿：",
            CLASS_NAMES[
                true_label
            ]
        )

        print(
            "AI预测：",
            predicted_name
        )

        print(
            f"模型概率："
            f"{probability * 100:.2f}%"
        )


# ============================================================
# 10. 启动程序
# ============================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    window = PostureWindow()

    window.show()

    sys.exit(
        app.exec()
    )