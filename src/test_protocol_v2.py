import random

import numpy as np
import pandas as pd

from serial_protocol_v2 import (
    build_pressure_frame_v2,
    PressureFrameParserV2,
    FRAME_SIZE
)


# ============================================================
# 1. 加载模拟数据
# ============================================================

DATA_PATH = (
    "data/simulated/"
    "posture_dataset_v2.csv"
)


df = pd.read_csv(
    DATA_PATH
)


# ============================================================
# 2. 创建解析器
# ============================================================

parser = PressureFrameParserV2()


print()
print("=" * 60)

print(
    "协议V2测试"
)

print(
    "每帧长度：",
    FRAME_SIZE,
    "bytes"
)

print("=" * 60)


# ============================================================
# 3. 连续发送20帧
# ============================================================

for sequence in range(20):

    random_index = np.random.randint(
        0,
        len(df)
    )


    row = df.iloc[
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


    # --------------------------------------------------------
    # 创建V2数据帧
    # --------------------------------------------------------

    frame = build_pressure_frame_v2(
        pressure,
        sequence
    )


    # --------------------------------------------------------
    # 模拟真实串口：
    #
    # 把一整帧拆成随机小块
    # --------------------------------------------------------

    position = 0


    while position < len(frame):

        chunk_size = random.randint(
            1,
            25
        )


        chunk = frame[
            position:
            position + chunk_size
        ]


        position += len(
            chunk
        )


        frames = parser.feed(
            chunk
        )


        for parsed in frames:

            received_pressure = (
                parsed["pressure"]
            )


            received_sequence = (
                parsed["sequence"]
            )


            error = np.max(
                np.abs(
                    pressure
                    -
                    received_pressure
                )
            )


            print(
                f"Seq:"
                f"{received_sequence:04d}"
                f" | "
                f"最大误差:"
                f"{error:.3f}"
            )


# ============================================================
# 4. 统计
# ============================================================

print()
print("=" * 60)

print(
    "正确帧：",
    parser.good_frames
)

print(
    "CRC错误：",
    parser.crc_errors
)

print(
    "帧尾错误：",
    parser.tail_errors
)

print(
    "检测到丢帧：",
    parser.lost_frames
)

print("=" * 60)