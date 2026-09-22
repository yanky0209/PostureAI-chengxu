import sys
import random

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
    QHBoxLayout,
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
# 1. 基本配置
# ============================================================

MODEL_PATH = "models/posture_cnn_v2.pt"

DATA_PATH = "data/simulated/posture_dataset_v2.csv"

# 10Hz = 每100ms更新一次
UPDATE_INTERVAL_MS = 100

# 每隔多少帧更换一次“模拟人的坐姿”
# 30帧 × 100ms = 3秒
POSTURE_CHANGE_FRAMES = 30


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
# 3. CNN结构
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
# 4. 加载CNN模型
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
# 5. 加载模拟数据
# ============================================================

df = pd.read_csv(
    DATA_PATH
)


# ============================================================
# 6. 把数据按类别分开
#
# 以后模拟某一个连续坐姿时，
# 不会每100ms突然从左倾跳到后仰
# ============================================================

class_data = {}

for label in range(5):

    class_data[label] = df[
        df["label"] == label
    ].reset_index(drop=True)


# ============================================================
# 7. CNN预测函数
# ============================================================

def predict_posture(pressure_matrix):

    pressure_matrix = np.asarray(
        pressure_matrix,
        dtype=np.float32
    )

    if pressure_matrix.shape != (8, 8):

        raise ValueError(
            "压力矩阵必须为8×8"
        )


    pressure_matrix = np.clip(
        pressure_matrix,
        0,
        100
    )


    normalized = (
        pressure_matrix
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

        output = model(tensor)

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
# 8. 压力热力图组件
# ============================================================

class PressureCanvas(FigureCanvasQTAgg):

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
# 9. GUI主窗口
# ============================================================

class PostureWindow(QMainWindow):

    def __init__(self):

        super().__init__()


        self.setWindowTitle(
            "PostureAI 智能坐姿监测系统"
        )

        self.resize(
            1100,
            700
        )


        # ====================================================
        # 实时监测变量
        # ====================================================

        self.is_monitoring = False

        self.frame_count = 0

        self.current_simulated_label = 0


        # ====================================================
        # QTimer
        # ====================================================

        self.timer = QTimer(self)

        self.timer.timeout.connect(
            self.update_frame
        )


        # ====================================================
        # 主区域
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
        # 中部
        # ====================================================

        content_layout = QHBoxLayout()


        # ----------------------------------------------------
        # 左边热力图
        # ----------------------------------------------------

        self.canvas = PressureCanvas()

        content_layout.addWidget(
            self.canvas,
            2
        )


        # ----------------------------------------------------
        # 右边信息
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


        self.frequency_label = QLabel(
            "刷新频率：10 Hz"
        )

        self.frequency_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.frequency_label.setStyleSheet(
            """
            font-size: 17px;
            padding: 5px;
            color: gray;
            """
        )


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
            self.frequency_label
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
        # 按钮区域
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

        # --- 监测会话统计变量 ---
        self.session_start_time = None
        self.posture_counter = Counter()
        self.total_samples = 0


    # ========================================================
    # 10. 开始实时监测
    # ========================================================

    def start_monitoring(self):

        if self.is_monitoring:

            return


        self.is_monitoring = True

        self.frame_count = 0

        # --- 新增：初始化本次监测的统计数据 ---
        self.session_start_time = time.time()
        self.posture_counter.clear()
        self.total_samples = 0


        # 随机选择一种初始坐姿
        self.current_simulated_label = (
            random.randint(0, 4)
        )


        self.timer.start(
            UPDATE_INTERVAL_MS
        )


        self.status_label.setText(
            "● 正在实时监测"
        )

        self.status_label.setStyleSheet(
            """
            font-size: 23px;
            font-weight: bold;
            padding: 20px;
            color: green;
            """
        )


        print()
        print("实时监测已启动")
        print("刷新频率：10 Hz")


    # ========================================================
    # 11. 停止监测
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

        # --- 新增：计算时长并生成记录报告 ---
        duration = time.time() - self.session_start_time if getattr(self, 'session_start_time', None) else 0
        
        if getattr(self, 'total_samples', 0) > 0:
            self.generate_report(duration)
        else:
            QMessageBox.information(self, "提示", "监测时间过短，未采集到有效数据。")



    # ========================================================
    # 12. 每100ms运行一次
    # ========================================================

    def update_frame(self):

        self.frame_count += 1


        # ----------------------------------------------------
        # 每3秒模拟用户改变一次坐姿
        # ----------------------------------------------------

        if (
            self.frame_count
            % POSTURE_CHANGE_FRAMES
            == 0
        ):

            self.current_simulated_label = (
                random.randint(0, 4)
            )


        # ----------------------------------------------------
        # 从当前坐姿类别随机取一帧
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
        # CNN推理
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
        # 更新热力图
        # ----------------------------------------------------

        self.canvas.update_pressure(
            pressure
        )


        # ----------------------------------------------------
        # 更新GUI
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
        # 健康状态
        # ----------------------------------------------------

        if predicted_name == "normal":

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


        else:

            self.status_label.setText(
                "⚠ 请调整坐姿"
            )

            self.status_label.setStyleSheet(
                """
                font-size: 23px;
                font-weight: bold;
                padding: 20px;
                color: red;
                """
            )


        # ----------------------------------------------------
        # 调试信息
        # ----------------------------------------------------

        print(
            f"真实: "
            f"{CLASS_NAMES[true_label]:8s}"
            f" | "
            f"预测: "
            f"{predicted_name:8s}"
            f" | "
            f"{probability * 100:6.2f}%"
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


# ============================================================
# 13. 启动程序
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
# --- 新增：28，623，638，702，783，867行