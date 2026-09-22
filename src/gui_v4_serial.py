import sys
import random
from collections import deque, Counter

import numpy as np
import pandas as pd

import serial

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
# 导入我们上一课写好的串口协议
# ============================================================

from serial_protocol import (
    build_pressure_frame,
    PressureFrameParser
)


# ============================================================
# 1. 基本配置
# ============================================================

MODEL_PATH = "models/posture_cnn_v2.pt"

DATA_PATH = "data/simulated/posture_dataset_v2.csv"


# ------------------------------------------------------------
# UART配置
# ------------------------------------------------------------

SERIAL_URL = "loop://"

BAUDRATE = 115200


# ------------------------------------------------------------
# 10Hz
#
# 每100ms一帧
# ------------------------------------------------------------

UPDATE_INTERVAL_MS = 100


# ------------------------------------------------------------
# 模拟用户8秒改变一次坐姿
# ------------------------------------------------------------

POSTURE_CHANGE_FRAMES = 80


# ------------------------------------------------------------
# 防抖最近10帧
# ------------------------------------------------------------

SMOOTHING_WINDOW = 10


# ------------------------------------------------------------
# 最低Softmax概率
# ------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.60


# ------------------------------------------------------------
# 演示阶段连续3秒不良姿势报警
# ------------------------------------------------------------

BAD_POSTURE_ALARM_SECONDS = 3.0


# ============================================================
# 2. 坐姿类别
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
# 5. 加载CNN
# ============================================================

model = PostureCNN().to(
    device
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ============================================================
# 6. AI预测函数
# ============================================================

def predict_posture(
    pressure_matrix
):

    pressure = np.asarray(
        pressure_matrix,
        dtype=np.float32
    )


    if pressure.shape != (8, 8):

        raise ValueError(
            "压力矩阵必须是8×8"
        )


    pressure = np.clip(
        pressure,
        0,
        100
    )


    # 与训练保持一致：
    #
    # 0~100
    # ↓
    # 0~1

    pressure = (
        pressure
        /
        100.0
    )


    tensor = torch.tensor(
        pressure,
        dtype=torch.float32
    )


    # 8×8
    # ↓
    # 1×1×8×8

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(
        device
    )


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
# 7. 加载模拟数据
#
# 注意：
#
# GUI不会直接使用这些数据。
#
# 它们只属于“假STM32”。
# ============================================================

df = pd.read_csv(
    DATA_PATH
)


class_data = {}

for label in range(5):

    class_data[label] = df[
        df["label"] == label
    ].reset_index(
        drop=True
    )


# ============================================================
# 8. 假STM32
# ============================================================

class FakeSTM32:

    def __init__(
        self,
        serial_port
    ):

        self.serial_port = (
            serial_port
        )

        self.frame_count = 0

        self.current_label = (
            random.randint(
                0,
                4
            )
        )


    # ========================================================
    # 每调用一次：
    #
    # 模拟STM32发送一帧UART数据
    # ========================================================

    def send_one_frame(self):

        self.frame_count += 1


        # ----------------------------------------------------
        # 每8秒改变一次真实坐姿
        # ----------------------------------------------------

        if (
            self.frame_count
            %
            POSTURE_CHANGE_FRAMES
            == 0
        ):

            old_label = (
                self.current_label
            )


            choices = [
                label
                for label in range(5)
                if label != old_label
            ]


            self.current_label = (
                random.choice(
                    choices
                )
            )


            print()

            print(
                "【假STM32】用户改变坐姿：",
                CLASS_NAMES[
                    self.current_label
                ]
            )


        # ----------------------------------------------------
        # 当前类别的数据
        # ----------------------------------------------------

        current_df = class_data[
            self.current_label
        ]


        random_index = (
            np.random.randint(
                0,
                len(current_df)
            )
        )


        row = current_df.iloc[
            random_index
        ]


        pressure = (
            row
            .drop(
                labels=["label"]
            )
            .values
            .astype(
                np.float32
            )
            .reshape(
                8,
                8
            )
        )


        # ----------------------------------------------------
        # 8×8压力矩阵
        #
        # ↓
        #
        # 0xAA
        # + 64×uint16
        # + 0xBB
        # ----------------------------------------------------

        frame = build_pressure_frame(
            pressure
        )


        # ----------------------------------------------------
        # 假装STM32 UART发送
        # ----------------------------------------------------

        self.serial_port.write(
            frame
        )


        self.serial_port.flush()


        return self.current_label


# ============================================================
# 9. Matplotlib热力图
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


        self.ax = (
            self.figure
            .add_subplot(111)
        )


        empty = np.zeros(
            (8, 8)
        )


        self.image = self.ax.imshow(
            empty,
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
# 10. GUI
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
            "PostureAI 串口实时坐姿监测系统"
        )

        self.resize(
            1180,
            740
        )


        # ====================================================
        # 打开假串口
        #
        # timeout=0
        #
        # 表示非阻塞读取
        # 避免GUI卡死
        # ====================================================

        self.serial_port = (
            serial.serial_for_url(
                SERIAL_URL,
                baudrate=BAUDRATE,
                timeout=0
            )
        )


        # ====================================================
        # 创建假STM32
        # ====================================================

        self.fake_stm32 = FakeSTM32(
            self.serial_port
        )


        # ====================================================
        # PC端串口解析器
        # ====================================================

        self.parser = (
            PressureFrameParser()
        )


        # ====================================================
        # 防抖
        # ====================================================

        self.prediction_history = deque(
            maxlen=SMOOTHING_WINDOW
        )


        self.stable_posture = None

        self.bad_posture_seconds = 0.0

        self.is_monitoring = False


        # ====================================================
        # QTimer
        # ====================================================

        self.timer = QTimer(
            self
        )


        self.timer.timeout.connect(
            self.serial_update
        )


        # ====================================================
        # 主窗口
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
        # 内容区域
        # ====================================================

        content_layout = QHBoxLayout()


        # ====================================================
        # 左侧热力图
        # ====================================================

        self.canvas = (
            PressureCanvas()
        )


        content_layout.addWidget(
            self.canvas,
            2
        )


        # ====================================================
        # 右侧
        # ====================================================

        info_layout = QVBoxLayout()


        # ----------------------------------------------------
        # 串口状态
        # ----------------------------------------------------

        self.serial_label = QLabel(
            "串口：loop:// @ 115200"
        )


        self.serial_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )


        self.serial_label.setStyleSheet(
            """
            font-size: 17px;
            color: gray;
            padding: 5px;
            """
        )


        # ----------------------------------------------------
        # 稳定坐姿
        # ----------------------------------------------------

        stable_title = QLabel(
            "稳定坐姿"
        )


        stable_title.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )


        stable_title.setStyleSheet(
            """
            font-size: 21px;
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
            font-size: 38px;
            font-weight: bold;
            padding: 15px;
            """
        )


        # ----------------------------------------------------
        # 单帧结果
        # ----------------------------------------------------

        self.raw_label = QLabel(
            "单帧预测：--"
        )


        self.raw_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )


        self.raw_label.setStyleSheet(
            """
            font-size: 18px;
            color: gray;
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
        # 串口帧数量
        # ----------------------------------------------------

        self.frame_label = QLabel(
            "已解析串口帧：0"
        )


        self.frame_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )


        self.frame_label.setStyleSheet(
            """
            font-size: 17px;
            color: gray;
            """
        )


        # ----------------------------------------------------
        # 不良时间
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
        # 状态
        # ----------------------------------------------------

        self.status_label = QLabel(
            "系统等待启动"
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
            self.serial_label
        )

        info_layout.addWidget(
            stable_title
        )

        info_layout.addWidget(
            self.posture_label
        )

        info_layout.addWidget(
            self.raw_label
        )

        info_layout.addWidget(
            self.probability_label
        )

        info_layout.addWidget(
            self.frame_label
        )

        info_layout.addWidget(
            self.duration_label
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

        button_layout = QHBoxLayout()


        self.start_button = QPushButton(
            "开始串口监测"
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


        # ====================================================
        # 已成功解析的帧数
        # ====================================================

        self.parsed_frame_count = 0


    # ========================================================
    # 11. 开始
    # ========================================================

    def start_monitoring(self):

        if self.is_monitoring:

            return


        self.is_monitoring = True


        self.prediction_history.clear()

        self.stable_posture = None

        self.bad_posture_seconds = 0.0

        self.parsed_frame_count = 0


        # 清空旧串口数据
        self.serial_port.reset_input_buffer()


        self.timer.start(
            UPDATE_INTERVAL_MS
        )


        self.status_label.setText(
            "● 串口实时监测中"
        )


        self.status_label.setStyleSheet(
            """
            font-size: 22px;
            font-weight: bold;
            padding: 20px;
            color: green;
            """
        )


        print()
        print("=" * 65)
        print("GUI V4 串口监测启动")
        print("串口：loop://")
        print("波特率：115200")
        print("采样率：10Hz")
        print("=" * 65)


    # ========================================================
    # 12. 停止
    # ========================================================

    def stop_monitoring(self):

        self.timer.stop()

        self.is_monitoring = False


        self.status_label.setText(
            "监测已停止"
        )


        self.status_label.setStyleSheet(
            """
            font-size: 22px;
            font-weight: bold;
            padding: 20px;
            color: gray;
            """
        )


        print()
        print("串口监测已停止")


    # ========================================================
    # 13. 多数投票
    # ========================================================

    def get_stable_posture(self):

        if len(
            self.prediction_history
        ) < SMOOTHING_WINDOW:

            return None


        counter = Counter(
            self.prediction_history
        )


        posture, _ = (
            counter.most_common(1)[0]
        )


        return posture


    # ========================================================
    # 14. 健康状态
    # ========================================================

    def update_health_status(
        self,
        stable_posture
    ):

        if stable_posture is None:

            self.bad_posture_seconds = 0.0


            self.duration_label.setText(
                "不良坐姿持续：0.0 秒"
            )


            self.status_label.setText(
                "正在分析坐姿..."
            )

            return


        # ----------------------------------------------------
        # 正常
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
                font-size: 22px;
                font-weight: bold;
                padding: 20px;
                color: green;
                """
            )

            return


        # ----------------------------------------------------
        # 不良姿势
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
                font-size: 22px;
                font-weight: bold;
                padding: 20px;
                color: orange;
                """
            )


        else:

            self.status_label.setText(
                f"⚠ "
                f"{CHINESE_NAMES[stable_posture]}"
                f"时间过长，请调整坐姿"
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
    # 15. 处理一帧8×8压力矩阵
    # ========================================================

    def process_pressure_frame(
        self,
        pressure
    ):

        self.parsed_frame_count += 1


        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        (
            predicted_index,
            predicted_name,
            probability

        ) = predict_posture(
            pressure
        )


        # ----------------------------------------------------
        # 热力图
        # ----------------------------------------------------

        self.canvas.update_pressure(
            pressure
        )


        # ----------------------------------------------------
        # 单帧结果
        # ----------------------------------------------------

        self.raw_label.setText(
            f"单帧预测："
            f"{CHINESE_NAMES[predicted_name]}"
        )


        self.probability_label.setText(
            f"模型概率："
            f"{probability * 100:.2f}%"
        )


        self.frame_label.setText(
            f"已解析串口帧："
            f"{self.parsed_frame_count}"
        )


        # ----------------------------------------------------
        # 概率合格才进入防抖
        # ----------------------------------------------------

        if (
            probability
            >=
            CONFIDENCE_THRESHOLD
        ):

            self.prediction_history.append(
                predicted_name
            )


        stable_posture = (
            self.get_stable_posture()
        )


        self.stable_posture = (
            stable_posture
        )


        # ----------------------------------------------------
        # 稳定坐姿显示
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


        self.update_health_status(
            stable_posture
        )


        # ----------------------------------------------------
        # 调试
        # ----------------------------------------------------

        stable_text = (
            stable_posture
            if stable_posture
            is not None
            else "..."
        )


        print(
            f"UART帧:"
            f"{self.parsed_frame_count:05d}"
            f" | "
            f"模拟真实:"
            f"{CLASS_NAMES[self.fake_stm32.current_label]:8s}"
            f" | "
            f"预测:"
            f"{predicted_name:8s}"
            f" | "
            f"稳定:"
            f"{stable_text:8s}"
            f" | "
            f"{probability * 100:6.2f}%"
        )


    # ========================================================
    # 16. 每100ms执行
    # ========================================================

    def serial_update(self):

        # ====================================================
        # A. 假STM32发送UART帧
        # ====================================================

        self.fake_stm32.send_one_frame()


        # ====================================================
        # B. PC从串口读取字节
        # ====================================================

        waiting = (
            self.serial_port.in_waiting
        )


        if waiting <= 0:

            return


        raw_data = (
            self.serial_port.read(
                waiting
            )
        )


        # ====================================================
        # C. 字节流进入协议解析器
        # ====================================================

        pressure_frames = (
            self.parser.feed(
                raw_data
            )
        )


        # ====================================================
        # D. 有可能一次读取到多帧
        #
        # 所以使用for
        # ====================================================

        for pressure in pressure_frames:

            self.process_pressure_frame(
                pressure
            )


    # ========================================================
    # 17. 窗口关闭
    # ========================================================

    def closeEvent(
        self,
        event
    ):

        self.timer.stop()


        if self.serial_port.is_open:

            self.serial_port.close()


        event.accept()


# ============================================================
# 18. 启动
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