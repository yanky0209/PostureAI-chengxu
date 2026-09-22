import numpy as np

from calibration import (
    kg_to_model_pressure,
    model_pressure_to_kg,
    adc_to_model_pressure_linear
)


# ============================================================
# 测试1：
# kg → 0～100
# ============================================================

pressure_kg = np.zeros(
    (8, 8),
    dtype=np.float32
)


pressure_kg[3, 3] = 1.0

pressure_kg[3, 4] = 2.5

pressure_kg[4, 3] = 4.0

pressure_kg[4, 4] = 5.0


model_pressure = (
    kg_to_model_pressure(
        pressure_kg
    )
)


print()
print("=" * 60)

print(
    "kg → CNN压力值"
)

print("=" * 60)

print(
    np.round(
        model_pressure,
        1
    )
)


# ============================================================
# 测试2：
# 0～100 → kg
# ============================================================

restored_kg = (
    model_pressure_to_kg(
        model_pressure
    )
)


print()
print("=" * 60)

print(
    "CNN压力值 → kg"
)

print("=" * 60)

print(
    np.round(
        restored_kg,
        2
    )
)


# ============================================================
# 测试3：
# 假设一个ADC例子
#
# 注意：
# 这里只测试程序，不代表真实硬件参数！
#
# 假设：
#
# 空载ADC = 500
# 满量程ADC = 3500
# ============================================================

fake_adc = np.full(
    (8, 8),
    2000,
    dtype=np.float32
)


adc_pressure = (
    adc_to_model_pressure_linear(
        fake_adc,
        zero_adc=500,
        full_scale_adc=3500
    )
)


print()
print("=" * 60)

print(
    "假ADC测试"
)

print(
    "注意：500和3500只是程序测试值，"
    "不是实际传感器标定结果。"
)

print("=" * 60)

print(
    np.round(
        adc_pressure,
        1
    )
)