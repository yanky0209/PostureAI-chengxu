import sys
import random
from collections import deque, Counter

import numpy as np
import pandas as pd

import serial
from serial.tools import list_ports

import torch
import torch.nn as nn

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QMessageBox
)

from PyQt6.QtCore import (
    Qt,
    QTimer
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


# --- 新增：时间 ---
import time
from collections import Counter
from config import CHINESE_NAMES


# ============================================================
# 导入协议 V2
# ============================================================

from serial_protocol_v2 import (
    build_pressure_frame_v2,
    PressureFrameParserV2
)


# ============================================================
# 1. 基本配置
# ============================================================

MODEL_PATH = "models/posture_cnn_v2.pt"

DATA_PATH = (
    "data/simulated/"
    "posture_dataset_v2.csv"
)


# ============================================================
# 2. 系统参数
# ============================================================

UPDATE_INTERVAL_MS = 100

POSTURE_CHANGE_FRAMES = 80

SMOOTHING_WINDOW = 10

CONFIDENCE_THRESHOLD = 0.60

BAD_POSTURE_ALARM_SECONDS = 3.0


# ============================================================
# 3. 坐姿类别
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
# 4. CPU / GPU
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 5. CNN
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
# 6. 加载模型
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
# 7. CNN预测
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


    # --------------------------------------------------------
    # 当前CNN训练输入：
    #
    # 0～100
    # ↓
    # 0～1
    # --------------------------------------------------------

    normalized = (
        pressure
        /
        100.0
    )


    tensor = torch.tensor(
        normalized,
        dtype=torch.float32
    )


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
# 8. 模拟数据集
#
# 只给 Fake STM32 使用
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
# 9. 假 STM32 V2
# ============================================================

class FakeSTM32:

    def __init__(
        self,
        serial_port
    ):

        self.serial_port = (
            serial_port
        )

        self.sequence = 0

        self.frame_count = 0

        self.current_label = (
            random.randint(
                0,
                4
            )
        )


    def send_one_frame(self):

        self.frame_count += 1


        # ----------------------------------------------------
        # 每8秒更换一种模拟姿势
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
                x
                for x in range(5)
                if x != old_label
            ]


            self.current_label = (
                random.choice(
                    choices
                )
            )


            print()

            print(
                "【Fake STM32】"
                "模拟用户换姿势：",
                CLASS_NAMES[
                    self.current_label
                ]
            )


        # ----------------------------------------------------
        # 取当前姿势数据
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
        # 协议V2打包
        # ----------------------------------------------------

        frame = build_pressure_frame_v2(
            pressure,
            self.sequence
        )


        # ----------------------------------------------------
        # UART发送
        # ----------------------------------------------------

        self.serial_port.write(
            frame
        )

        self.serial_port.flush()


        # ----------------------------------------------------
        # Seq + 1
        #
        # uint16溢出回0
        # ----------------------------------------------------

        self.sequence = (
            self.sequence
            +
            1
        ) & 0xFFFF


# ============================================================
# 10. 压力热力图
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
# 11. 主窗口
# ============================================================

class PostureWindow(QMainWindow):

    def __init__(self):

        super().__init__()


        # ====================================================
        # 窗口
        # ====================================================

        self.setWindowTitle(
            "PostureAI 智能坐姿监测系统 V5"
        )


        self.resize(
            1250,
            800
        )


        # ====================================================
        # 串口对象
        # ====================================================

        self.serial_port = None

        self.fake_stm32 = None


        # ====================================================
        # 连接状态
        # ====================================================

        self.is_connected = False

        self.is_monitoring = False


        # ====================================================
        # 串口协议解析器
        # ====================================================

        self.parser = (
            PressureFrameParserV2()
        )


        # ====================================================
        # 防抖
        # ====================================================

        self.prediction_history = deque(
            maxlen=SMOOTHING_WINDOW
        )


        self.stable_posture = None

        self.bad_posture_seconds = 0.0


        # ====================================================
        # 定时器
        # ====================================================

        self.timer = QTimer(
            self
        )


        self.timer.timeout.connect(
            self.serial_update
        )


        # ====================================================
        # 主界面
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
            padding: 12px;
            """
        )


        main_layout.addWidget(
            title
        )


        # ====================================================
        # 设备控制区域
        # ====================================================

        device_layout = QHBoxLayout()


        # ----------------------------------------------------
        # 数据源
        # ----------------------------------------------------

        mode_title = QLabel(
            "数据源："
        )


        self.mode_combo = QComboBox()


        self.mode_combo.addItems(
            [
                "模拟模式",
                "真实串口"
            ]
        )


        self.mode_combo.currentIndexChanged.connect(
            self.mode_changed
        )


        # ----------------------------------------------------
        # COM口
        # ----------------------------------------------------

        port_title = QLabel(
            "串口："
        )


        self.port_combo = QComboBox()


        # ----------------------------------------------------
        # 刷新COM
        # ----------------------------------------------------

        self.refresh_button = QPushButton(
            "刷新串口"
        )


        self.refresh_button.clicked.connect(
            self.refresh_ports
        )


        # ----------------------------------------------------
        # 波特率
        # ----------------------------------------------------

        baud_title = QLabel(
            "波特率："
        )


        self.baud_combo = QComboBox()


        self.baud_combo.addItems(
            [
                "9600",
                "57600",
                "115200"
            ]
        )


        self.baud_combo.setCurrentText(
            "115200"
        )


        # ----------------------------------------------------
        # 连接
        # ----------------------------------------------------

        self.connect_button = QPushButton(
            "连接设备"
        )


        self.connect_button.clicked.connect(
            self.connect_device
        )


        # ----------------------------------------------------
        # 断开
        # ----------------------------------------------------

        self.disconnect_button = QPushButton(
            "断开设备"
        )


        self.disconnect_button.clicked.connect(
            self.disconnect_device
        )


        # ----------------------------------------------------
        # 放入布局
        # ----------------------------------------------------

        device_layout.addWidget(
            mode_title
        )

        device_layout.addWidget(
            self.mode_combo
        )

        device_layout.addWidget(
            port_title
        )

        device_layout.addWidget(
            self.port_combo
        )

        device_layout.addWidget(
            self.refresh_button
        )

        device_layout.addWidget(
            baud_title
        )

        device_layout.addWidget(
            self.baud_combo
        )

        device_layout.addWidget(
            self.connect_button
        )

        device_layout.addWidget(
            self.disconnect_button
        )


        main_layout.addLayout(
            device_layout
        )


        # ====================================================
        # 连接状态
        # ====================================================

        self.connection_label = QLabel(
            "● 未连接"
        )


        self.connection_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )


        self.connection_label.setStyleSheet(
            """
            font-size: 17px;
            font-weight: bold;
            color: gray;
            padding: 5px;
            """
        )


        main_layout.addWidget(
            self.connection_label
        )


        # ====================================================
        # 主内容区域
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
            font-size: 40px;
            font-weight: bold;
            padding: 15px;
            """
        )


        # ----------------------------------------------------
        # 单帧预测
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
        # 概率
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
        # Seq
        # ----------------------------------------------------

        self.sequence_label = QLabel(
            "最新帧序号：--"
        )


        self.sequence_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )


        self.sequence_label.setStyleSheet(
            """
            font-size: 17px;
            color: gray;
            """
        )


        # ----------------------------------------------------
        # 通信统计
        # ----------------------------------------------------

        self.communication_label = QLabel(
            "正确帧：0 | CRC错误：0 | 丢帧：0"
        )


        self.communication_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )


        self.communication_label.setStyleSheet(
            """
            font-size: 16px;
            color: gray;
            padding: 5px;
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
        # 健康状态
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
            self.sequence_label
        )

        info_layout.addWidget(
            self.communication_label
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


        # --- 新增：监测会话统计变量 ---
        
        self.session_start_time = None
        self.posture_counter = Counter()
        self.total_samples = 0


        # ====================================================
        # 开始/停止监测
        # ====================================================

        monitor_layout = QHBoxLayout()


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


        monitor_layout.addWidget(
            self.start_button
        )

        monitor_layout.addWidget(
            self.stop_button
        )


        main_layout.addLayout(
            monitor_layout
        )


        # ====================================================
        # 初始化COM列表
        # ====================================================

        self.refresh_ports()

        self.mode_changed()


    # ========================================================
    # 12. 数据源模式切换
    # ========================================================

    def mode_changed(self):

        mode = (
            self.mode_combo.currentText()
        )


        if mode == "模拟模式":

            self.port_combo.setEnabled(
                False
            )

            self.refresh_button.setEnabled(
                False
            )


        else:

            self.port_combo.setEnabled(
                True
            )

            self.refresh_button.setEnabled(
                True
            )


    # ========================================================
    # 13. 自动扫描COM口
    # ========================================================

    def refresh_ports(self):

        self.port_combo.clear()


        ports = list(
            list_ports.comports()
        )


        if not ports:

            self.port_combo.addItem(
                "未检测到串口"
            )

            return


        for port in ports:

            # ------------------------------------------------
            # 用户看到：
            #
            # COM6 - 蓝牙链接上的标准串行
            # ------------------------------------------------

            text = (
                f"{port.device}"
                f" - "
                f"{port.description}"
            )


            # ------------------------------------------------
            # userData保存真正的COM名称
            #
            # 比如COM6
            # ------------------------------------------------

            self.port_combo.addItem(
                text,
                port.device
            )


    # ========================================================
    # 14. 连接设备
    # ========================================================

    def connect_device(self):

        if self.is_connected:

            QMessageBox.information(
                self,
                "提示",
                "设备已经连接。"
            )

            return


        mode = (
            self.mode_combo.currentText()
        )


        baudrate = int(
            self.baud_combo.currentText()
        )


        try:

            # =================================================
            # 模拟模式
            # =================================================

            if mode == "模拟模式":

                self.serial_port = (
                    serial.serial_for_url(
                        "loop://",
                        baudrate=baudrate,
                        timeout=0
                    )
                )


                self.fake_stm32 = FakeSTM32(
                    self.serial_port
                )


                connection_text = (
                    "● 已连接：模拟 STM32"
                )


            # =================================================
            # 真实串口模式
            # =================================================

            else:

                port_name = (
                    self.port_combo
                    .currentData()
                )


                if not port_name:

                    QMessageBox.warning(
                        self,
                        "串口错误",
                        "没有可用的真实串口。"
                    )

                    return


                self.serial_port = (
                    serial.Serial(
                        port=port_name,
                        baudrate=baudrate,
                        bytesize=8,
                        parity="N",
                        stopbits=1,
                        timeout=0
                    )
                )


                self.fake_stm32 = None


                connection_text = (
                    f"● 已连接："
                    f"{port_name}"
                    f" @ "
                    f"{baudrate}"
                )


            # =================================================
            # 新建解析器
            #
            # 统计全部清零
            # =================================================

            self.parser = (
                PressureFrameParserV2()
            )


            self.is_connected = True


            self.connection_label.setText(
                connection_text
            )


            self.connection_label.setStyleSheet(
                """
                font-size: 17px;
                font-weight: bold;
                color: green;
                padding: 5px;
                """
            )


            # 连接之后不允许随便换模式
            self.mode_combo.setEnabled(
                False
            )

            self.port_combo.setEnabled(
                False
            )

            self.baud_combo.setEnabled(
                False
            )

            self.refresh_button.setEnabled(
                False
            )


            print()
            print("=" * 65)

            print(
                "设备连接成功"
            )

            print(
                "模式：",
                mode
            )

            print(
                "波特率：",
                baudrate
            )

            print("=" * 65)


        except Exception as error:

            QMessageBox.critical(
                self,
                "连接失败",
                str(error)
            )


            self.serial_port = None

            self.fake_stm32 = None

            self.is_connected = False


    # ========================================================
    # 15. 断开设备
    # ========================================================

    def disconnect_device(self):

        # ----------------------------------------------------
        # 先停止监测
        # ----------------------------------------------------

        self.stop_monitoring()


        # ----------------------------------------------------
        # 关闭串口
        # ----------------------------------------------------

        if (
            self.serial_port
            is not None
        ):

            try:

                if self.serial_port.is_open:

                    self.serial_port.close()

            except Exception:

                pass


        self.serial_port = None

        self.fake_stm32 = None

        self.is_connected = False


        self.connection_label.setText(
            "● 未连接"
        )


        self.connection_label.setStyleSheet(
            """
            font-size: 17px;
            font-weight: bold;
            color: gray;
            padding: 5px;
            """
        )


        # ----------------------------------------------------
        # 恢复控件
        # ----------------------------------------------------

        self.mode_combo.setEnabled(
            True
        )

        self.baud_combo.setEnabled(
            True
        )


        self.mode_changed()


        print()
        print("设备已断开")


    # ========================================================
    # 16. 开始监测
    # ========================================================

    def start_monitoring(self):

        if not self.is_connected:

            QMessageBox.warning(
                self,
                "未连接设备",
                "请先点击“连接设备”。"
            )

            return


        if self.is_monitoring:

            return


        self.is_monitoring = True


        self.prediction_history.clear()

        self.stable_posture = None

        self.bad_posture_seconds = 0.0

        # --- 新增：初始化本次监测的统计数据 ---
        self.session_start_time = time.time()
        self.posture_counter.clear()
        self.total_samples = 0



        # ----------------------------------------------------
        # 新的解析器
        #
        # 每次重新开始，统计重新计数
        # ----------------------------------------------------

        self.parser = (
            PressureFrameParserV2()
        )


        # ----------------------------------------------------
        # 清除旧串口数据
        # ----------------------------------------------------

        try:

            self.serial_port.reset_input_buffer()

        except Exception:

            pass


        self.timer.start(
            UPDATE_INTERVAL_MS
        )


        self.status_label.setText(
            "● 正在实时监测"
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
        print("实时监测启动")


    # ========================================================
    # 17. 停止监测
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

        # --- 新增：计算时长并生成记录报告 ---
        duration = time.time() - self.session_start_time if getattr(self, 'session_start_time', None) else 0
        
        if getattr(self, 'total_samples', 0) > 0:
            self.generate_report(duration)
        else:
            QMessageBox.information(self, "提示", "监测时间过短，未采集到有效数据。")


    # ========================================================
    # 18. 多数投票
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
    # 19. 更新健康状态
    # ========================================================

    def update_health_status(
        self,
        stable_posture
    ):

        # ----------------------------------------------------
        # 还在分析
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
                font-size: 22px;
                font-weight: bold;
                padding: 20px;
                color: orange;
                """
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
        # 不良坐姿
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
        # 尚未达到报警时间
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
                font-size: 22px;
                font-weight: bold;
                padding: 20px;
                color: orange;
                """
            )


        # ----------------------------------------------------
        # 正式提醒
        # ----------------------------------------------------

        else:

            chinese_name = (
                CHINESE_NAMES[
                    stable_posture
                ]
            )


            self.status_label.setText(
                f"⚠ "
                f"{chinese_name}"
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
    # 20. 处理一帧有效压力数据
    # ========================================================

    def process_frame(
        self,
        sequence,
        pressure
    ):

        # ----------------------------------------------------
        # AI
        # ----------------------------------------------------

        (
            predicted_index,
            predicted_name,
            probability

        ) = predict_posture(
            pressure
        )

        # --- 新增：记录当前监测数据 ---
        if self.is_monitoring:
            self.posture_counter[predicted_name] += 1
            self.total_samples += 1


        # ----------------------------------------------------
        # 热力图
        # ----------------------------------------------------

        self.canvas.update_pressure(
            pressure
        )


        # ----------------------------------------------------
        # 单帧坐姿
        # ----------------------------------------------------

        self.raw_label.setText(
            f"单帧预测："
            f"{CHINESE_NAMES[predicted_name]}"
        )


        # ----------------------------------------------------
        # 模型概率
        # ----------------------------------------------------

        self.probability_label.setText(
            f"模型概率："
            f"{probability * 100:.2f}%"
        )


        # ----------------------------------------------------
        # Seq
        # ----------------------------------------------------

        self.sequence_label.setText(
            f"最新帧序号："
            f"{sequence}"
        )


        # ----------------------------------------------------
        # 通信统计
        # ----------------------------------------------------

        self.communication_label.setText(
            f"正确帧："
            f"{self.parser.good_frames}"
            f" | "
            f"CRC错误："
            f"{self.parser.crc_errors}"
            f" | "
            f"丢帧："
            f"{self.parser.lost_frames}"
        )


        # ----------------------------------------------------
        # 置信度合格才加入防抖
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
        # 稳定结果
        # ----------------------------------------------------

        stable_posture = (
            self.get_stable_posture()
        )


        self.stable_posture = (
            stable_posture
        )


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
        # 健康提醒
        # ----------------------------------------------------

        self.update_health_status(
            stable_posture
        )


        # ----------------------------------------------------
        # 控制台输出
        # ----------------------------------------------------

        stable_text = (
            stable_posture
            if stable_posture
            is not None
            else "..."
        )


        print(
            f"Seq:"
            f"{sequence:05d}"
            f" | "
            f"预测:"
            f"{predicted_name:8s}"
            f" | "
            f"稳定:"
            f"{stable_text:8s}"
            f" | "
            f"{probability * 100:6.2f}%"
            f" | "
            f"CRC:"
            f"{self.parser.crc_errors}"
            f" | "
            f"丢帧:"
            f"{self.parser.lost_frames}"
        )

    # ----------------------------------------------------
    # 新增：生成弹窗并保存历史
    # ----------------------------------------------------
    def generate_report(self, duration):
        minutes, seconds = int(duration // 60), int(duration % 60)
        
        report_text = f"<b>【本次坐姿监测报告】</b><br><br>"
        report_text += f"⏱️ <b>监测时长：</b> {minutes}分{seconds}秒<br>"
        report_text += f"📊 <b>采样总数：</b> {self.total_samples}帧<br><br>"
        report_text += "<b>坐姿分布比例：</b><br>"
        
        most_common_label = self.posture_counter.most_common(1)[0][0]
        
        for label, count in self.posture_counter.items():
            percentage = (count / self.total_samples) * 100
            name_cn = CHINESE_NAMES.get(label, label)
            report_text += f"- {name_cn}: <b>{percentage:.1f}%</b><br>"
        
        main_cn = CHINESE_NAMES.get(most_common_label, most_common_label)
        report_text += f"<br>💡 <b>健康建议：</b>本次主要处于 <b>{main_cn}</b> 状态。"
        if most_common_label != 'normal':
            report_text += "建议适当调整坐姿并起身活动。"

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("监测结束 - 坐姿记录")
        msg_box.setText(report_text)
        msg_box.exec()

        self.save_history(duration, main_cn)

    def save_history(self, duration, main_posture):
        import csv
        from datetime import datetime
        import os
        
        filename = "session_history.csv"
        file_exists = os.path.exists(filename)

        with open(filename, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["时间", "时长(秒)", "主要坐姿", "正常占比(%)"])
            
            normal_count = self.posture_counter.get('normal', 0)
            normal_ratio = (normal_count / self.total_samples) * 100
            start_str = datetime.fromtimestamp(self.session_start_time).strftime('%Y-%m-%d %H:%M:%S')
            
            writer.writerow([start_str, round(duration, 1), main_posture, round(normal_ratio, 1)])



    # ========================================================
    # 21. 每100ms执行
    # ========================================================

    def serial_update(self):

        if (
            not self.is_connected
            or
            self.serial_port is None
        ):

            return


        try:

            # =================================================
            # 模拟模式：
            #
            # Fake STM32先发送一帧
            # =================================================

            if (
                self.mode_combo.currentText()
                ==
                "模拟模式"
            ):

                if (
                    self.fake_stm32
                    is not None
                ):

                    self.fake_stm32.send_one_frame()


            # =================================================
            # 看串口有多少字节
            # =================================================

            waiting = (
                self.serial_port.in_waiting
            )


            if waiting <= 0:

                return


            # =================================================
            # 读取串口字节
            # =================================================

            raw_data = (
                self.serial_port.read(
                    waiting
                )
            )


            # =================================================
            # 进入协议V2解析器
            # =================================================

            parsed_frames = (
                self.parser.feed(
                    raw_data
                )
            )


            # =================================================
            # 一次可能解析出多帧
            # =================================================

            for frame in parsed_frames:

                sequence = (
                    frame["sequence"]
                )


                pressure = (
                    frame["pressure"]
                )


                self.process_frame(
                    sequence,
                    pressure
                )


            # -------------------------------------------------
            # 即使CRC错误导致没有有效帧，
            # 统计仍然要更新
            # -------------------------------------------------

            self.communication_label.setText(
                f"正确帧："
                f"{self.parser.good_frames}"
                f" | "
                f"CRC错误："
                f"{self.parser.crc_errors}"
                f" | "
                f"丢帧："
                f"{self.parser.lost_frames}"
            )


        except Exception as error:

            print(
                "串口读取错误：",
                error
            )


            self.timer.stop()

            self.is_monitoring = False


            QMessageBox.critical(
                self,
                "串口读取失败",
                str(error)
            )


    # ========================================================
    # 22. 关闭窗口
    # ========================================================

    def closeEvent(
        self,
        event
    ):

        self.timer.stop()


        if (
            self.serial_port
            is not None
        ):

            try:

                if self.serial_port.is_open:

                    self.serial_port.close()

            except Exception:

                pass


        event.accept()


# ============================================================
# 23. 启动程序
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
# --- 新增：33、1084、1539、1619、1820、1966