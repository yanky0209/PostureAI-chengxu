import struct
import numpy as np


# ============================================================
# 1. 协议配置
# ============================================================

HEADER = 0xAA
TAIL = 0xBB

ROWS = 8
COLS = 8

SENSOR_COUNT = 64

BYTES_PER_SENSOR = 2

PRESSURE_SCALE = 10.0


# ============================================================
# 2. 长度
# ============================================================

SEQ_SIZE = 2

PAYLOAD_SIZE = (
    SENSOR_COUNT
    *
    BYTES_PER_SENSOR
)

CRC_SIZE = 2


# ------------------------------------------------------------
# 1帧：
#
# HEADER       1
# SEQ          2
# PRESSURE   128
# CRC          2
# TAIL         1
#
# 总共 134 bytes
# ------------------------------------------------------------

FRAME_SIZE = (
    1
    +
    SEQ_SIZE
    +
    PAYLOAD_SIZE
    +
    CRC_SIZE
    +
    1
)


# ============================================================
# 3. CRC16 MODBUS
# ============================================================

def crc16_modbus(data):

    crc = 0xFFFF

    for byte in data:

        crc ^= byte

        for _ in range(8):

            if crc & 0x0001:

                crc >>= 1
                crc ^= 0xA001

            else:

                crc >>= 1

    return crc & 0xFFFF


# ============================================================
# 4. 压力矩阵 → 128 bytes
# ============================================================

def encode_pressure_matrix(
    pressure_matrix
):

    pressure = np.asarray(
        pressure_matrix,
        dtype=np.float32
    )


    if pressure.shape != (
        ROWS,
        COLS
    ):

        raise ValueError(
            "压力矩阵必须是8×8"
        )


    pressure = np.clip(
        pressure,
        0,
        100
    )


    values = np.rint(
        pressure
        *
        PRESSURE_SCALE
    ).astype(
        np.uint16
    )


    # 左上角开始逐行扫描
    values = values.reshape(
        -1
    )


    payload = struct.pack(
        "<64H",
        *values.tolist()
    )


    return payload


# ============================================================
# 5. 创建 V2 完整数据帧
# ============================================================

def build_pressure_frame_v2(
    pressure_matrix,
    sequence
):

    # --------------------------------------------------------
    # sequence限制为uint16
    # --------------------------------------------------------

    sequence = (
        int(sequence)
        &
        0xFFFF
    )


    seq_bytes = struct.pack(
        "<H",
        sequence
    )


    payload = encode_pressure_matrix(
        pressure_matrix
    )


    # --------------------------------------------------------
    # CRC覆盖：
    #
    # Seq + 128字节压力数据
    #
    # 不包括AA和BB
    # --------------------------------------------------------

    crc_data = (
        seq_bytes
        +
        payload
    )


    crc_value = crc16_modbus(
        crc_data
    )


    crc_bytes = struct.pack(
        "<H",
        crc_value
    )


    frame = (
        bytes([HEADER])
        +
        seq_bytes
        +
        payload
        +
        crc_bytes
        +
        bytes([TAIL])
    )


    return frame


# ============================================================
# 6. 解码压力
# ============================================================

def decode_pressure_payload(
    payload
):

    if len(payload) != PAYLOAD_SIZE:

        raise ValueError(
            "压力payload长度错误"
        )


    values = struct.unpack(
        "<64H",
        payload
    )


    values = np.array(
        values,
        dtype=np.float32
    )


    values = (
        values
        /
        PRESSURE_SCALE
    )


    pressure = values.reshape(
        ROWS,
        COLS
    )


    return pressure


# ============================================================
# 7. V2帧解析器
# ============================================================

class PressureFrameParserV2:

    def __init__(self):

        self.buffer = bytearray()

        self.good_frames = 0

        self.crc_errors = 0

        self.tail_errors = 0

        self.lost_frames = 0

        self.last_sequence = None


    # ========================================================
    # 检查帧序号
    # ========================================================

    def check_sequence(
        self,
        sequence
    ):

        if self.last_sequence is None:

            self.last_sequence = (
                sequence
            )

            return


        expected = (
            self.last_sequence
            +
            1
        ) & 0xFFFF


        if sequence != expected:

            # ------------------------------------------------
            # 计算丢失了多少帧
            # ------------------------------------------------

            difference = (
                sequence
                -
                expected
            ) & 0xFFFF


            # 避免乱序/异常帧造成巨大数字
            if difference < 1000:

                self.lost_frames += (
                    difference
                )


        self.last_sequence = (
            sequence
        )


    # ========================================================
    # 喂串口数据
    # ========================================================

    def feed(
        self,
        data
    ):

        if data:

            self.buffer.extend(
                data
            )


        frames = []


        while True:

            # ------------------------------------------------
            # 找0xAA
            # ------------------------------------------------

            header_position = (
                self.buffer.find(
                    bytes([HEADER])
                )
            )


            if header_position == -1:

                self.buffer.clear()

                break


            # ------------------------------------------------
            # 丢掉AA之前的垃圾
            # ------------------------------------------------

            if header_position > 0:

                del self.buffer[
                    :header_position
                ]


            # ------------------------------------------------
            # 数据不够一整帧
            # ------------------------------------------------

            if len(
                self.buffer
            ) < FRAME_SIZE:

                break


            # ------------------------------------------------
            # 检查BB
            # ------------------------------------------------

            if (
                self.buffer[
                    FRAME_SIZE - 1
                ]
                != TAIL
            ):

                self.tail_errors += 1

                del self.buffer[0]

                continue


            # ------------------------------------------------
            # 取完整帧
            # ------------------------------------------------

            frame = bytes(
                self.buffer[
                    :FRAME_SIZE
                ]
            )


            # ------------------------------------------------
            # Seq
            #
            # byte 1~2
            # ------------------------------------------------

            sequence = struct.unpack(
                "<H",
                frame[
                    1:3
                ]
            )[0]


            # ------------------------------------------------
            # pressure payload
            #
            # byte 3 ~ 130
            # ------------------------------------------------

            payload_start = 3

            payload_end = (
                payload_start
                +
                PAYLOAD_SIZE
            )


            payload = frame[
                payload_start:
                payload_end
            ]


            # ------------------------------------------------
            # 收到的CRC
            # ------------------------------------------------

            received_crc = (
                struct.unpack(
                    "<H",
                    frame[
                        payload_end:
                        payload_end + 2
                    ]
                )[0]
            )


            # ------------------------------------------------
            # 自己重新计算CRC
            # ------------------------------------------------

            crc_data = frame[
                1:
                payload_end
            ]


            calculated_crc = (
                crc16_modbus(
                    crc_data
                )
            )


            # ------------------------------------------------
            # CRC错误
            # ------------------------------------------------

            if (
                received_crc
                !=
                calculated_crc
            ):

                self.crc_errors += 1

                del self.buffer[0]

                continue


            # ------------------------------------------------
            # 解码压力
            # ------------------------------------------------

            pressure = (
                decode_pressure_payload(
                    payload
                )
            )


            # ------------------------------------------------
            # 检查序号
            # ------------------------------------------------

            self.check_sequence(
                sequence
            )


            self.good_frames += 1


            frames.append(
                {
                    "sequence": sequence,
                    "pressure": pressure
                }
            )


            # ------------------------------------------------
            # 删除这一帧
            # ------------------------------------------------

            del self.buffer[
                :FRAME_SIZE
            ]


        return frames