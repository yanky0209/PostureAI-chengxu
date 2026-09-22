import sys
import random
from collections import deque, Counter

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

from PyQt6.QtCore import (
    Qt,
    QTimer
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


# ============================================================
# 1. 基本配置
# ============================================================

MODEL_PATH = "models/posture_cnn_v2.pt"

DATA_PATH = "data/simulated/posture_dataset_v2.csv"


# ------------------------------------------------------------
# 采样/刷新频率
# 100 ms = 10 Hz
# ------------------------------------------------------------

UPDATE_INTERVAL_MS = 100


# ------------------------------------------------------------
# 模拟用户每8秒改变一次坐姿
#
# 80帧 × 100ms = 8秒
# ------------------------------------------------------------

POSTURE_CHANGE_FRAMES = 80


# ------------------------------------------------------------
# 防抖窗口
#
# 最近10帧投票
# 10帧 × 100ms = 1秒
# ------------------------------------------------------------

SMOOTHING_WINDOW = 10


# ------------------------------------------------------------
# AI最低概率阈值
#
# 小于60%的预测暂时不加入投票
# ------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.60


# ------------------------------------------------------------
# 不良坐姿持续多少秒以后报警
# ------------------------------------------------------------

BAD_POSTURE_ALARM_SECONDS = 3.0


# ============================================================
# 2. 类别名称
# ============================================================

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
# 3. CPU / GPU
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 4. CNN模型
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
# 5. 加载CNN模型
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
# 6. 加载模拟数据
# ============================================================

df = pd.read_csv(
    DATA_PATH
)


# ============================================================
# 7. 按类别拆分数据
# ============================================================

class_data = {}

for label in range(5):

    class_data[label] = df[
        df["label"] == label
    ].reset_index(drop=True)


# ============================================================
# 8. CNN预测函数
# ============================================================

def predict_posture(
    pressure_matrix
):

    pressure_matrix = np.asarray(
        pressure_matrix,
        dtype=np.float32
    )


    if pressure_matrix.shape != (8, 8):

        raise ValueError(
            "压力矩阵必须是8×8"
        )


    pressure_matrix = np.clip(
        pressure_matrix,
        0,
        100
    )


    # --------------------------------------------------------
    # 和训练时一样：
    # 0~100 → 0~1
    # --------------------------------------------------------

    normalized = (
        pressure_matrix
        /
        100.0
    )


    tensor = torch.tensor(
        normalized,
        dtype=torch.float32
    )


    # 8×8
    # ↓
    # 1×1×8×8

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)


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


    return (
        predicted_index,
        predicted_name,
        probability
    )


# ============================================================
# 9. 热力图组件
# ============================================================

class PressureCanvas(
    FigureCanvasQTAgg
):

    def __init__(self):

        self.figure = Figure(
            figsize=(6, 6)
        )

        super().__init__(
            self.figure
        )


        self.ax = self.figure.add_subplot(
            111
        )


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
# 10. GUI主窗口
# ============================================================

class PostureWindow(
    QMainWindow
):

    def __init__(self):

        super().__init__()


        # ====================================================
        # 窗口
        # ====================================================

        self.setWindowTitle(
            "PostureAI 智能坐姿监测系统 V3"
        )

        self.resize(
            1150,
            720
        )


        # ====================================================
        # 监测状态
        # ====================================================

        self.is_monitoring = False

        self.frame_count = 0

        self.current_simulated_label = 0


        # ====================================================
        # 防抖预测历史
        #
        # 最多保存最近10帧
        # ====================================================

        self.prediction_history = deque(
            maxlen=SMOOTHING_WINDOW
        )


        # ====================================================
        # 稳定坐姿
        # ====================================================

        self.stable_posture = None


        # ====================================================
        # 不良坐姿累计时间
        # ====================================================

        self.bad_posture_seconds = 0.0


        # ====================================================
        # QTimer
        # ====================================================

        self.timer = QTimer(self)

        self.timer.timeout.connect(
            self.update_frame
        )


        # ====================================================
        # 主窗口布局
        # ====================================================

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
            font-size: 30px;
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


        # ====================================================
        # 左侧热力图
        # ====================================================

        self.canvas = PressureCanvas()

        content_layout.addWidget(
            self.canvas,
            2
        )


        # ====================================================
        # 右侧信息栏
        # ====================================================

        info_layout = QVBoxLayout()


        # ----------------------------------------------------
        # 当前坐姿标题
        # ----------------------------------------------------

        current_title = QLabel(
            "稳定坐姿"
        )

        current_title.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        current_title.setStyleSheet(
            """
            font-size: 21px;
            font-weight: bold;
            """
        )


        # ----------------------------------------------------
        # 稳定坐姿
        # ----------------------------------------------------

        self.posture_label = QLabel(
            "等待检测"
        )

        self.posture_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.posture_label.setStyleSheet(
            """
            font-size: 38px;
            font-weight: bold;
            padding: 15px;
            """
        )


        # ----------------------------------------------------
        # 当前单帧预测
        # ----------------------------------------------------

        self.raw_prediction_label = QLabel(
            "单帧预测：--"
        )

        self.raw_prediction_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.raw_prediction_label.setStyleSheet(
            """
            font-size: 17px;
            color: gray;
            padding: 5px;
            """
        )


        # ----------------------------------------------------
        # 模型概率
        # ----------------------------------------------------

        self.probability_label = QLabel(
            "模型概率：--"
        )

        self.probability_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.probability_label.setStyleSheet(
            """
            font-size: 19px;
            padding: 5px;
            """
        )


        # ----------------------------------------------------
        # 防抖窗口
        # ----------------------------------------------------

        self.smoothing_label = QLabel(
            "防抖窗口：10帧"
        )

        self.smoothing_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.smoothing_label.setStyleSheet(
            """
            font-size: 16px;
            color: gray;
            """
        )


        # ----------------------------------------------------
        # 持续时间
        # ----------------------------------------------------

        self.duration_label = QLabel(
            "不良坐姿持续：0.0 秒"
        )

        self.duration_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.duration_label.setStyleSheet(
            """
            font-size: 19px;
            padding: 10px;
            """
        )


        # ----------------------------------------------------
        # 系统状态
        # ----------------------------------------------------

        self.status_label = QLabel(
            "系统等待启动"
        )

        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.status_label.setStyleSheet(
            """
            font-size: 23px;
            font-weight: bold;
            padding: 20px;
            """
        )


        # ----------------------------------------------------
        # 采样率
        # ----------------------------------------------------

        self.frequency_label = QLabel(
            "刷新频率：10 Hz"
        )

        self.frequency_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.frequency_label.setStyleSheet(
            """
            font-size: 16px;
            color: gray;
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
            self.raw_prediction_label
        )

        info_layout.addWidget(
            self.probability_label
        )

        info_layout.addWidget(
            self.smoothing_label
        )

        info_layout.addWidget(
            self.duration_label
        )

        info_layout.addWidget(
            self.status_label
        )

        info_layout.addWidget(
            self.frequency_label
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

        button_layout = QHBoxLayout()


        self.start_button = QPushButton(
            "开始监测"
        )

        self.stop_button = QPushButton(
            "停止监测"
        )


        self.start_button.setMinimumHeight(
            55
        )

        self.stop_button.setMinimumHeight(
            55
        )


        self.start_button.setStyleSheet(
            """
            font-size: 20px;
            font-weight: bold;
            """
        )

        self.stop_button.setStyleSheet(
            """
            font-size: 20px;
            font-weight: bold;
            """
        )


        self.start_button.clicked.connect(
            self.start_monitoring
        )

        self.stop_button.clicked.connect(
            self.stop_monitoring
        )


        button_layout.addWidget(
            self.start_button
        )

        button_layout.addWidget(
            self.stop_button
        )


        main_layout.addLayout(
            button_layout
        )


    # ========================================================
    # 11. 开始监测
    # ========================================================

    def start_monitoring(self):

        if self.is_monitoring:

            return


        self.is_monitoring = True

        self.frame_count = 0

        self.bad_posture_seconds = 0.0

        self.stable_posture = None

        self.prediction_history.clear()


        self.current_simulated_label = (
            random.randint(
                0,
                4
            )
        )


        self.timer.start(
            UPDATE_INTERVAL_MS
        )


        self.status_label.setText(
            "● 正在建立稳定判断..."
        )


        self.status_label.setStyleSheet(
            """
            font-size: 23px;
            font-weight: bold;
            padding: 20px;
            color: orange;
            """
        )


        print()
        print("=" * 60)
        print("PostureAI V3 实时监测启动")
        print("采样频率：10 Hz")
        print("防抖窗口：10帧")
        print("报警阈值：连续不良坐姿3秒")
        print("=" * 60)


    # ========================================================
    # 12. 停止监测
    # ========================================================

    def stop_monitoring(self):

        self.timer.stop()

        self.is_monitoring = False


        self.status_label.setText(
            "监测已停止"
        )


        self.status_label.setStyleSheet(
            """
            font-size: 23px;
            font-weight: bold;
            padding: 20px;
            color: gray;
            """
        )


        print()
        print("实时监测已停止")


    # ========================================================
    # 13. 多数投票
    # ========================================================

    def get_stable_posture(self):

        # ----------------------------------------------------
        # 预测数量不足
        # ----------------------------------------------------

        if len(
            self.prediction_history
        ) < SMOOTHING_WINDOW:

            return None


        # ----------------------------------------------------
        # 统计最近10帧
        # ----------------------------------------------------

        counter = Counter(
            self.prediction_history
        )


        # ----------------------------------------------------
        # 找出现次数最多的姿势
        # ----------------------------------------------------

        posture, count = (
            counter.most_common(1)[0]
        )


        return posture


    # ========================================================
    # 14. 更新健康提醒
    # ========================================================

    def update_health_status(
        self,
        stable_posture
    ):

        # ----------------------------------------------------
        # 还没有形成稳定判断
        # ----------------------------------------------------

        if stable_posture is None:

            self.bad_posture_seconds = 0.0


            self.duration_label.setText(
                "不良坐姿持续：0.0 秒"
            )


            self.status_label.setText(
                "正在分析坐姿..."
            )


            self.status_label.setStyleSheet(
                """
                font-size: 23px;
                font-weight: bold;
                padding: 20px;
                color: orange;
                """
            )

            return


        # ----------------------------------------------------
        # 正常坐姿
        # ----------------------------------------------------

        if stable_posture == "normal":

            self.bad_posture_seconds = 0.0


            self.duration_label.setText(
                "不良坐姿持续：0.0 秒"
            )


            self.status_label.setText(
                "✓ 坐姿正常"
            )


            self.status_label.setStyleSheet(
                """
                font-size: 23px;
                font-weight: bold;
                padding: 20px;
                color: green;
                """
            )

            return


        # ----------------------------------------------------
        # 不良坐姿
        #
        # 每一帧增加0.1秒
        # ----------------------------------------------------

        self.bad_posture_seconds += (
            UPDATE_INTERVAL_MS
            /
            1000.0
        )


        self.duration_label.setText(
            f"不良坐姿持续："
            f"{self.bad_posture_seconds:.1f} 秒"
        )


        # ----------------------------------------------------
        # 还没有达到报警阈值
        # ----------------------------------------------------

        if (
            self.bad_posture_seconds
            <
            BAD_POSTURE_ALARM_SECONDS
        ):

            self.status_label.setText(
                "正在观察姿势..."
            )


            self.status_label.setStyleSheet(
                """
                font-size: 23px;
                font-weight: bold;
                padding: 20px;
                color: orange;
                """
            )


        # ----------------------------------------------------
        # 达到报警阈值
        # ----------------------------------------------------

        else:

            chinese_name = CHINESE_NAMES[
                stable_posture
            ]


            self.status_label.setText(
                f"⚠ {chinese_name}时间过长，请调整坐姿"
            )


            self.status_label.setStyleSheet(
                """
                font-size: 21px;
                font-weight: bold;
                padding: 20px;
                color: red;
                """
            )


    # ========================================================
    # 15. 每100ms更新一次
    # ========================================================

    def update_frame(self):

        self.frame_count += 1


        # ----------------------------------------------------
        # 每8秒模拟用户换姿势
        # ----------------------------------------------------

        if (
            self.frame_count
            %
            POSTURE_CHANGE_FRAMES
            == 0
        ):

            old_label = (
                self.current_simulated_label
            )


            choices = [
                x
                for x in range(5)
                if x != old_label
            ]


            self.current_simulated_label = (
                random.choice(
                    choices
                )
            )


            print()
            print(
                "模拟用户改变坐姿：",
                CLASS_NAMES[
                    self.current_simulated_label
                ]
            )


        # ----------------------------------------------------
        # 取当前坐姿的一条数据
        # ----------------------------------------------------

        current_df = class_data[
            self.current_simulated_label
        ]


        random_index = np.random.randint(
            0,
            len(current_df)
        )


        row = current_df.iloc[
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
        # CNN预测
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
        # 显示单帧AI结果
        # ----------------------------------------------------

        self.raw_prediction_label.setText(
            f"单帧预测："
            f"{CHINESE_NAMES[predicted_name]}"
        )


        self.probability_label.setText(
            f"模型概率："
            f"{probability * 100:.2f}%"
        )


        # ----------------------------------------------------
        # 置信度合格才进入防抖历史
        # ----------------------------------------------------

        if (
            probability
            >=
            CONFIDENCE_THRESHOLD
        ):

            self.prediction_history.append(
                predicted_name
            )


        # ----------------------------------------------------
        # 获得稳定坐姿
        # ----------------------------------------------------

        stable_posture = (
            self.get_stable_posture()
        )


        self.stable_posture = (
            stable_posture
        )


        # ----------------------------------------------------
        # GUI显示稳定坐姿
        # ----------------------------------------------------

        if stable_posture is None:

            self.posture_label.setText(
                "分析中..."
            )


        else:

            self.posture_label.setText(
                CHINESE_NAMES[
                    stable_posture
                ]
            )


        # ----------------------------------------------------
        # 健康状态
        # ----------------------------------------------------

        self.update_health_status(
            stable_posture
        )


        # ----------------------------------------------------
        # 调试输出
        # ----------------------------------------------------

        stable_text = (
            stable_posture
            if stable_posture is not None
            else "..."
        )


        print(
            f"真实:"
            f"{CLASS_NAMES[true_label]:8s}"
            f" | "
            f"单帧:"
            f"{predicted_name:8s}"
            f" | "
            f"稳定:"
            f"{stable_text:8s}"
            f" | "
            f"{probability * 100:6.2f}%"
            f" | "
            f"持续:"
            f"{self.bad_posture_seconds:4.1f}s"
        )


# ============================================================
# 16. 启动
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