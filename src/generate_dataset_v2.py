import os
import numpy as np
import pandas as pd


# ============================================================
# 基本配置
# ============================================================

ROWS = 8
COLS = 8

SAMPLES_PER_CLASS = 1500

SAVE_DIR = "data/simulated"
SAVE_PATH = os.path.join(
    SAVE_DIR,
    "posture_dataset_v2.csv"
)


# ============================================================
# 标签定义
# ============================================================

POSTURES = {
    "normal": 0,
    "left": 1,
    "right": 2,
    "forward": 3,
    "backward": 4
}


# ============================================================
# 生成二维高斯压力峰
# ============================================================

def gaussian_2d(
    x,
    y,
    center_x,
    center_y,
    sigma_x,
    sigma_y
):

    return np.exp(
        -(
            ((x - center_x) ** 2)
            / (2 * sigma_x ** 2)
            +
            ((y - center_y) ** 2)
            / (2 * sigma_y ** 2)
        )
    )


# ============================================================
# 生成单个模拟坐姿
# ============================================================

def generate_posture(posture):

    # --------------------------------------------------------
    # 建立 8×8 坐标
    # --------------------------------------------------------

    y, x = np.mgrid[
        0:ROWS,
        0:COLS
    ]


    # --------------------------------------------------------
    # 模拟不同人体整体压力
    #
    # 暂时使用归一化压力，不直接用 kg
    # 后面真实硬件数据进来以后再根据 ADC 做标定
    # --------------------------------------------------------

    body_scale = np.random.uniform(
        0.75,
        1.20
    )


    # --------------------------------------------------------
    # 人坐在坐垫上的随机位置偏差
    # --------------------------------------------------------

    random_x = np.random.uniform(
        -0.30,
        0.30
    )

    random_y = np.random.uniform(
        -0.30,
        0.30
    )


    # --------------------------------------------------------
    # 默认左右臀中心位置
    #
    # 这里暂定：
    # X 小 = 左侧
    # X 大 = 右侧
    #
    # Y 小 = 坐垫前侧
    # Y 大 = 坐垫后侧
    # --------------------------------------------------------

    left_x = 2.4 + random_x
    right_x = 4.6 + random_x

    center_y = 4.0 + random_y


    # --------------------------------------------------------
    # 两个臀部压力峰基础强度
    # --------------------------------------------------------

    left_strength = np.random.uniform(
        65,
        85
    )

    right_strength = np.random.uniform(
        65,
        85
    )


    # ========================================================
    # 不同坐姿改变压力分布
    # ========================================================

    if posture == "normal":

        # 正常情况下允许轻微左右不平衡
        imbalance = np.random.uniform(
            0.90,
            1.10
        )

        left_strength *= imbalance
        right_strength /= imbalance


    elif posture == "left":

        # 左倾程度随机
        severity = np.random.uniform(
            0.15,
            0.45
        )

        left_strength *= (
            1 + severity
        )

        right_strength *= (
            1 - severity
        )

        # 压力中心稍向左移动
        left_x -= np.random.uniform(
            0.10,
            0.45
        )

        right_x -= np.random.uniform(
            0.05,
            0.30
        )


    elif posture == "right":

        severity = np.random.uniform(
            0.15,
            0.45
        )

        right_strength *= (
            1 + severity
        )

        left_strength *= (
            1 - severity
        )

        left_x += np.random.uniform(
            0.05,
            0.30
        )

        right_x += np.random.uniform(
            0.10,
            0.45
        )


    elif posture == "forward":

        # 暂定 Y 小的一端是坐垫前侧
        shift = np.random.uniform(
            0.35,
            0.90
        )

        center_y -= shift


    elif posture == "backward":

        shift = np.random.uniform(
            0.35,
            0.90
        )

        center_y += shift


    else:

        raise ValueError(
            f"未知坐姿：{posture}"
        )


    # ========================================================
    # 每个人臀部压力范围略有区别
    # ========================================================

    sigma_x = np.random.uniform(
        0.85,
        1.20
    )

    sigma_y = np.random.uniform(
        0.95,
        1.35
    )


    # ========================================================
    # 左臀压力
    # ========================================================

    left_peak = (
        left_strength
        *
        gaussian_2d(
            x,
            y,
            left_x,
            center_y,
            sigma_x,
            sigma_y
        )
    )


    # ========================================================
    # 右臀压力
    # ========================================================

    right_peak = (
        right_strength
        *
        gaussian_2d(
            x,
            y,
            right_x,
            center_y,
            sigma_x,
            sigma_y
        )
    )


    # ========================================================
    # 合成整个压力矩阵
    # ========================================================

    pressure = (
        left_peak
        +
        right_peak
    )

    pressure *= body_scale


    # ========================================================
    # 加入随机零点漂移
    # ========================================================

    drift = np.random.uniform(
        -3,
        3
    )

    pressure += drift


    # ========================================================
    # 加入传感器随机噪声
    # ========================================================

    noise = np.random.normal(
        0,
        4,
        (ROWS, COLS)
    )

    pressure += noise


    # ========================================================
    # 随机模拟少量异常传感器
    # ========================================================

    if np.random.random() < 0.08:

        bad_y = np.random.randint(
            0,
            ROWS
        )

        bad_x = np.random.randint(
            0,
            COLS
        )

        pressure[
            bad_y,
            bad_x
        ] *= np.random.uniform(
            0.0,
            0.30
        )


    # ========================================================
    # 限制到 0～100
    #
    # 注意：
    # 这里的 0～100 是软件归一化值，
    # 不是直接代表 kg。
    # ========================================================

    pressure = np.clip(
        pressure,
        0,
        100
    )

    return pressure


# ============================================================
# 创建数据集
# ============================================================

dataset = []

print()
print("=" * 60)
print("开始生成 V2 模拟压力数据集")
print("=" * 60)


for posture, label in POSTURES.items():

    print(
        f"正在生成：{posture}"
    )

    for _ in range(
        SAMPLES_PER_CLASS
    ):

        pressure = generate_posture(
            posture
        )

        features = pressure.flatten()

        row = list(
            features
        )

        row.append(
            label
        )

        dataset.append(
            row
        )


# ============================================================
# 创建列名
# ============================================================

columns = [
    f"sensor_{i}"
    for i in range(64)
]

columns.append(
    "label"
)


# ============================================================
# 转成 DataFrame
# ============================================================

df = pd.DataFrame(
    dataset,
    columns=columns
)


# ============================================================
# 随机打乱
# ============================================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)


# ============================================================
# 保存
# ============================================================

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

df.to_csv(
    SAVE_PATH,
    index=False
)


# ============================================================
# 输出统计
# ============================================================

print()
print("=" * 60)
print("V2 数据集生成完成！")
print("=" * 60)

print(
    "总样本数：",
    len(df)
)

print(
    "传感器特征数：",
    64
)

print(
    "类别数量：",
    len(POSTURES)
)

print(
    "保存位置：",
    SAVE_PATH
)

print()

print(
    "类别统计："
)

print(
    df["label"]
    .value_counts()
    .sort_index()
)

print()

print(
    "标签对应关系："
)

for posture, label in POSTURES.items():

    print(
        label,
        "=",
        posture
    )