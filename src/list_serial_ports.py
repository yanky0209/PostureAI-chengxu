from serial.tools import list_ports


ports = list(
    list_ports.comports()
)


print()
print("=" * 50)
print("当前检测到的串口")
print("=" * 50)


if not ports:

    print(
        "没有检测到真实串口设备。"
    )


else:

    for port in ports:

        print()
        print(
            "设备：",
            port.device
        )

        print(
            "名称：",
            port.description
        )

        print(
            "硬件ID：",
            port.hwid
        )


print()
print("=" * 50)