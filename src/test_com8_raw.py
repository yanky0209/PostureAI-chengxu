import serial
import time


PORT = "COM8"
BAUDRATE = 115200


print("=" * 60)
print("PostureAI STM32 串口原始数据测试")
print("=" * 60)

print(f"准备打开：{PORT}")
print(f"波特率：{BAUDRATE}")


ser = serial.Serial(
    port=PORT,
    baudrate=BAUDRATE,
    bytesize=8,
    parity="N",
    stopbits=1,
    timeout=1
)


print()
print("串口打开成功！")
print("正在等待 STM32 数据……")
print()


try:

    while True:

        waiting = ser.in_waiting

        if waiting > 0:

            data = ser.read(waiting)

            print(
                f"收到 {len(data)} Byte："
            )

            print(
                data.hex(" ").upper()
            )

            print("-" * 60)

        else:

            print(
                "暂时没有收到数据..."
            )

        time.sleep(1)


except KeyboardInterrupt:

    print()
    print("停止测试")


finally:

    ser.close()

    print(
        "COM8 已关闭"
    )