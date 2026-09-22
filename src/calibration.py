import numpy as np

from config import (
    SENSOR_ROWS,
    SENSOR_COLS,
    MAX_LOAD_PER_SENSOR_KG,
    MODEL_PRESSURE_MIN,
    MODEL_PRESSURE_MAX
)


# ============================================================
# 1. 检查压力矩阵尺寸
# ============================================================

def check_matrix(
    matrix
):

    matrix = np.asarray(
        matrix,
        dtype=np.float32
    )


    if matrix.shape != (
        SENSOR_ROWS,
        SENSOR_COLS
    ):

        raise ValueError(
            f"压力矩阵必须为 "
            f"{SENSOR_ROWS}×{SENSOR_COLS}"
        )


    return matrix


# ============================================================
# 2. kg → AI使用的0～100
# ============================================================

def kg_to_model_pressure(
    pressure_kg
):

    pressure_kg = check_matrix(
        pressure_kg
    )


    # --------------------------------------------------------
    # 限制在传感器额定量程内
    #
    # 小于0 → 0
    # 大于5kg → 5kg
    # --------------------------------------------------------

    pressure_kg = np.clip(
        pressure_kg,
        0.0,
        MAX_LOAD_PER_SENSOR_KG
    )


    # --------------------------------------------------------
    # 0～5kg
    #
    # ↓
    #
    # 0～1
    # --------------------------------------------------------

    normalized = (
        pressure_kg
        /
        MAX_LOAD_PER_SENSOR_KG
    )


    # --------------------------------------------------------
    # 0～1
    #
    # ↓
    #
    # 0～100
    # --------------------------------------------------------

    model_pressure = (
        normalized
        *
        MODEL_PRESSURE_MAX
    )


    return model_pressure.astype(
        np.float32
    )


# ============================================================
# 3. AI使用的0～100 → kg
#
# 主要用于以后GUI显示或调试
# ============================================================

def model_pressure_to_kg(
    model_pressure
):

    model_pressure = check_matrix(
        model_pressure
    )


    model_pressure = np.clip(
        model_pressure,
        MODEL_PRESSURE_MIN,
        MODEL_PRESSURE_MAX
    )


    pressure_kg = (
        model_pressure
        /
        MODEL_PRESSURE_MAX
        *
        MAX_LOAD_PER_SENSOR_KG
    )


    return pressure_kg.astype(
        np.float32
    )


# ============================================================
# 4. ADC → 0～100
#
# 简单线性标定模板
#
# ⚠ 现在不能认为这就是真实传感器标定公式。
#
# 等硬件到手以后，
# 必须实际测量：
#
# 空载ADC
# 已知重量ADC
#
# 然后再确定真实转换关系。
# ============================================================

def adc_to_model_pressure_linear(
    adc_matrix,
    zero_adc,
    full_scale_adc
):

    adc_matrix = check_matrix(
        adc_matrix
    )


    zero_adc = np.asarray(
        zero_adc,
        dtype=np.float32
    )


    full_scale_adc = np.asarray(
        full_scale_adc,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # 满量程ADC - 零点ADC
    # --------------------------------------------------------

    denominator = (
        full_scale_adc
        -
        zero_adc
    )


    # --------------------------------------------------------
    # 防止除以0
    # --------------------------------------------------------

    denominator = np.where(
        np.abs(denominator) < 1e-6,
        1.0,
        denominator
    )


    # --------------------------------------------------------
    # ADC
    #
    # ↓
    #
    # 0～1
    # --------------------------------------------------------

    normalized = (
        adc_matrix
        -
        zero_adc
    ) / denominator


    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )


    # --------------------------------------------------------
    # 0～1
    #
    # ↓
    #
    # 0～100
    # --------------------------------------------------------

    model_pressure = (
        normalized
        *
        MODEL_PRESSURE_MAX
    )


    return model_pressure.astype(
        np.float32
    )