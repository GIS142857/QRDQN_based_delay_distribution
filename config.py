# ---------------------------- time ------------------------------ #
UNIT = 1e6  # us

# ------------------------- net topology ------------------------- #
SUM_NODES = 18
SRC, DST = [0, 1, 2], [15, 16, 17]
FLOW_DICT = {
    0: 15,
    1: 16,
    2: 17,
}
ADJ_TABLE = {0: [3, 4],
             1: [5, 6],
             2: [4, 5],
             3: [9, 10],
             4: [8, 9],
             5: [6, 7, 8],
             6: [7],
             7: [12, 13],
             8: [9, 12],
             9: [10, 11],
             10: [15, 16],
             11: [16, 17],
             12: [11, 13, 14],
             13: [14],
             14: [17],
             15: [],
             16: [],
             17: [],
             }
NODE_POSITION = {
    0: [89, 372],
    1: [75, 70],
    2: [57, 215],
    3: [175, 412],
    4: [125, 295],
    5: [120, 154],
    6: [174, 54],
    7: [223, 120],
    8: [185, 252],
    9: [229, 333],
    10: [297, 399],
    11: [310, 284],
    12: [267, 197],
    13: [310, 92],
    14: [350, 160],
    15: [372, 447],
    16: [390, 347],
    17: [406, 242],
}

# -------------------------- net layer --------------------------- #
VIOLATE_P = 1e-3  # violate probability
MAX_RETRIES = 5  # maximum retransmission attempts
PACKET_HEADER_LENGTH = 128  # bit
PACKET_PAYLOAD_LENGTH = 1000 * 8  # bit
PACKET_LENGTH = PACKET_HEADER_LENGTH + PACKET_PAYLOAD_LENGTH
ARRIVAL_RATE = {  # packets arrival_rates of src_node (us)
    'flow1': 19500,
    'flow2': 12500,
    'flow3': 8192,
}
FLOW_MAP = {0: 'flow1', 1: 'flow2', 2: 'flow3'}
DEADLINE = {'flow1': 0.1 * UNIT, 'flow2': 0.2 * UNIT, 'flow3': 0.5 * UNIT}
DEFAULT_MAX_DELAY = DEADLINE['flow3']

# -------------------------- mac layer --------------------------- #
MAC_TYPE = 'TDMA'
FRAME_SLOT = {
    0: 1,
    1: 3,
    2: 2,
    3: 2,
    4: 3,
    5: 1,
    6: 4,
    7: 5,
    8: 6,
    9: 4,
    10: 5,
    11: 1,
    12: 2,
    13: 3,
    14: 4,
    15: 0,
    16: 0,
    17: 0,
}
SLOT_LIST = {  # slot: nodes
    1: [0, 5, 11],
    2: [2, 3, 12],
    3: [1, 4, 13],
    4: [6, 9, 14],
    5: [7, 10],
    6: [8],
}
SLOT_DURATION = 2000  # us
MAC_HEADER_LENGTH = 32  # 32 byte fixed fields of a mac packet
MAC_PAYLOAD_LENGTH = PACKET_LENGTH
MAC_PDU_LENGTH = MAC_HEADER_LENGTH + PACKET_PAYLOAD_LENGTH

# -------------------------- phy layer --------------------------- #
PHY_TYPE = 'Rayleigh-Fading Model'
ALPHA = 4
BETA = 1
LIGHT_SPEED = 3e8  # light speed (m/s)
BAND_WIDTH = 5 * UNIT  # HZ
NOISE_STRENGTH = -174.0  # env-noise power:-174 dBm/Hz
TRANSMITTING_POWER = 8  # dBm
GAIN_T = 1  # 发射天线增益
GAIN_R = 1  # 接收天线增益
FREQUENCY = 2.4e9  # Hz，信号频率 2.4 GHz
# BIT_RATE = BAND_WIDTH * Spectral_Efficiency (BPSK: 1 bps/Hz, QPSK: 2 bps/Hz, 16-QAM: 4bps/Hz, 64-QAM: 6 bps/Hz)
BIT_RATE = BAND_WIDTH * 1  # using BPSK
BIT_TRANSMISSION_TIME = 1 / BIT_RATE * 1e6  # us
BIT_TRANSPORT_TIME = 1 / 3e8 * UNIT  # bits transport time
PHY_HEADER_LENGTH = 32
PHY_PAYLOAD_LENGTH = MAC_HEADER_LENGTH + MAC_PAYLOAD_LENGTH
PHY_PDU_LENGTH = PHY_HEADER_LENGTH + PHY_PAYLOAD_LENGTH
# print(PHY_PDU_LENGTH)
# print(BIT_TRANSMISSION_TIME * PHY_PDU_LENGTH)
