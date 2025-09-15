# Optimized SM4 implementation with precomputed constants and lookup tables

import numpy as np

# Precomputed constants
S_BOX = None
FK = None
CK = None


def _initialize_constants():
    """Initialize precomputed constants once"""
    global S_BOX, FK, CK

    if S_BOX is None:
        # S-box from the original code
        s_hex = [
            [
                "D6",
                "90",
                "E9",
                "FE",
                "CC",
                "E1",
                "3D",
                "B7",
                "16",
                "B6",
                "14",
                "C2",
                "28",
                "FB",
                "2C",
                "05",
            ],
            [
                "2B",
                "67",
                "9A",
                "76",
                "2A",
                "BE",
                "04",
                "C3",
                "AA",
                "44",
                "13",
                "26",
                "49",
                "86",
                "06",
                "99",
            ],
            [
                "9C",
                "42",
                "50",
                "F4",
                "91",
                "EF",
                "98",
                "7A",
                "33",
                "54",
                "0B",
                "43",
                "ED",
                "CF",
                "AC",
                "62",
            ],
            [
                "E4",
                "B3",
                "1C",
                "A9",
                "C9",
                "08",
                "E8",
                "95",
                "80",
                "DF",
                "94",
                "FA",
                "75",
                "8F",
                "3F",
                "A6",
            ],
            [
                "47",
                "07",
                "A7",
                "FC",
                "F3",
                "73",
                "17",
                "BA",
                "83",
                "59",
                "3C",
                "19",
                "E6",
                "85",
                "4F",
                "A8",
            ],
            [
                "68",
                "6B",
                "81",
                "B2",
                "71",
                "64",
                "DA",
                "8B",
                "F8",
                "EB",
                "0F",
                "4B",
                "70",
                "56",
                "9D",
                "35",
            ],
            [
                "1E",
                "24",
                "0E",
                "5E",
                "63",
                "58",
                "D1",
                "A2",
                "25",
                "22",
                "7C",
                "3B",
                "01",
                "21",
                "78",
                "87",
            ],
            [
                "D4",
                "00",
                "46",
                "57",
                "9F",
                "D3",
                "27",
                "52",
                "4C",
                "36",
                "02",
                "E7",
                "A0",
                "C4",
                "C8",
                "9E",
            ],
            [
                "EA",
                "BF",
                "8A",
                "D2",
                "40",
                "C7",
                "38",
                "B5",
                "A3",
                "F7",
                "F2",
                "CE",
                "F9",
                "61",
                "15",
                "A1",
            ],
            [
                "E0",
                "AE",
                "5D",
                "A4",
                "9B",
                "34",
                "1A",
                "55",
                "AD",
                "93",
                "32",
                "30",
                "F5",
                "8C",
                "B1",
                "E3",
            ],
            [
                "1D",
                "F6",
                "E2",
                "2E",
                "82",
                "66",
                "CA",
                "60",
                "C0",
                "29",
                "23",
                "AB",
                "0D",
                "53",
                "4E",
                "6F",
            ],
            [
                "D5",
                "DB",
                "37",
                "45",
                "DE",
                "FD",
                "8E",
                "2F",
                "03",
                "FF",
                "6A",
                "72",
                "6D",
                "6C",
                "5B",
                "51",
            ],
            [
                "8D",
                "1B",
                "AF",
                "92",
                "BB",
                "DD",
                "BC",
                "7F",
                "11",
                "D9",
                "5C",
                "41",
                "1F",
                "10",
                "5A",
                "D8",
            ],
            [
                "0A",
                "C1",
                "31",
                "88",
                "A5",
                "CD",
                "7B",
                "BD",
                "2D",
                "74",
                "D0",
                "12",
                "B8",
                "E5",
                "B4",
                "B0",
            ],
            [
                "89",
                "69",
                "97",
                "4A",
                "0C",
                "96",
                "77",
                "7E",
                "65",
                "B9",
                "F1",
                "09",
                "C5",
                "6E",
                "C6",
                "84",
            ],
            [
                "18",
                "F0",
                "7D",
                "EC",
                "3A",
                "DC",
                "4D",
                "20",
                "79",
                "EE",
                "5F",
                "3E",
                "D7",
                "CB",
                "39",
                "48",
            ],
        ]

        # Convert to lookup table
        S_BOX = [0] * 256
        for i in range(16):
            for j in range(16):
                S_BOX[i * 16 + j] = int(s_hex[i][j], 16)

    if FK is None:
        FK = [0xA3B1BAC6, 0x56AA3350, 0x677D9197, 0xB27022DC]

    if CK is None:
        ck_hex = [
            "00070E15",
            "1C232A31",
            "383F464D",
            "545B6269",
            "70777E85",
            "8C939AA1",
            "A8AFB6BD",
            "C4CBD2D9",
            "E0E7EEF5",
            "FC030A11",
            "181F262D",
            "343B4249",
            "50575E65",
            "6C737A81",
            "888F969D",
            "A4ABB2B9",
            "C0C7CED5",
            "DCE3EAF1",
            "F8FF060D",
            "141B2229",
            "30373E45",
            "4C535A61",
            "686F767D",
            "848B9299",
            "A0A7AEB5",
            "BCC3CAD1",
            "D8DFE6ED",
            "F4FB0209",
            "10171E25",
            "2C333A41",
            "484F565D",
            "646B7279",
        ]
        CK = [int(x, 16) for x in ck_hex]


def rotl(x, n):
    """Left rotate a 32-bit integer"""
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


def s_box_substitution(x):
    """Apply S-box substitution to a 32-bit word"""
    _initialize_constants()
    result = 0
    for i in range(4):
        byte_val = (x >> (24 - 8 * i)) & 0xFF
        result = (result << 8) | S_BOX[byte_val]
    return result


def l_transformation(x):
    """Apply L transformation"""
    return x ^ rotl(x, 2) ^ rotl(x, 10) ^ rotl(x, 18) ^ rotl(x, 24)


def l_key_transformation(x):
    """Apply L' transformation for key expansion"""
    return x ^ rotl(x, 13) ^ rotl(x, 23)


def key_expansion(master_key):
    """Generate round keys from master key"""
    _initialize_constants()

    # Convert master key to 4 words
    mk = [0] * 4
    for i in range(4):
        mk[i] = 0
        for j in range(4):
            mk[i] = (mk[i] << 8) | master_key[i * 4 + j]

    # Key expansion step 1: MKi XOR FKi
    k = [mk[i] ^ FK[i] for i in range(4)]

    # Key expansion step 2: Generate round keys
    rk = [0] * 32
    for i in range(32):
        temp = k[1] ^ k[2] ^ k[3] ^ CK[i]
        temp = s_box_substitution(temp)
        temp = l_key_transformation(temp)
        rk[i] = k[0] ^ temp
        k.append(rk[i])
        k.pop(0)

    return rk


def round_function(x0, x1, x2, x3, rk):
    """SM4 round function"""
    temp = x1 ^ x2 ^ x3 ^ rk
    temp = s_box_substitution(temp)
    temp = l_transformation(temp)
    return x0 ^ temp


def sm4_encrypt_block(plaintext_block, round_keys, rounds=32):
    """Encrypt a single 128-bit block"""
    # Convert block to 4 words
    x = [0] * 4
    for i in range(4):
        x[i] = 0
        for j in range(4):
            x[i] = (x[i] << 8) | plaintext_block[i * 4 + j]

    # Apply round function
    for i in range(rounds):
        temp = round_function(x[0], x[1], x[2], x[3], round_keys[i])
        x = [x[1], x[2], x[3], temp]

    # Reverse final transformation
    result = [x[3], x[2], x[1], x[0]]

    # Convert back to bytes
    output = []
    for word in result:
        for i in range(4):
            output.append((word >> (24 - 8 * i)) & 0xFF)

    return output


def sm4_decrypt_block(ciphertext_block, round_keys, rounds=32):
    """Decrypt a single 128-bit block"""
    # Convert block to 4 words
    x = [0] * 4
    for i in range(4):
        x[i] = 0
        for j in range(4):
            x[i] = (x[i] << 8) | ciphertext_block[i * 4 + j]

    # Apply round function with reversed keys
    for i in range(rounds):
        temp = round_function(x[0], x[1], x[2], x[3], round_keys[rounds - 1 - i])
        x = [x[1], x[2], x[3], temp]

    # Reverse final transformation
    result = [x[3], x[2], x[1], x[0]]

    # Convert back to bytes
    output = []
    for word in result:
        for i in range(4):
            output.append((word >> (24 - 8 * i)) & 0xFF)

    return output


# Binary conversion utilities (similar to HIGHT)
def binary_to_bytes(binary_string):
    """Convert binary string to list of bytes"""
    binary_string = binary_string.replace(" ", "")
    if len(binary_string) % 8 != 0:
        raise ValueError(
            f"Binary string length must be multiple of 8, got {len(binary_string)}"
        )

    bytes_list = []
    for i in range(0, len(binary_string), 8):
        byte_str = binary_string[i : i + 8]
        bytes_list.append(int(byte_str, 2))
    return bytes_list


def bytes_to_binary(bytes_list):
    """Convert list of bytes to binary string"""
    binary_string = ""
    for byte in bytes_list:
        binary_string += format(byte, "08b")
    return binary_string


def integer_to_bytes(key_int):
    """Convert 128-bit integer key to list of 16 bytes"""
    bytes_list = []
    for i in range(16):
        bytes_list.append((key_int >> (8 * (15 - i))) & 0xFF)
    return bytes_list


def numpy_array_to_bytes(bit_array):
    """Convert numpy array of bits to list of bytes"""
    if len(bit_array) % 8 != 0:
        raise ValueError(
            f"Bit array length must be multiple of 8, got {len(bit_array)}"
        )

    bytes_list = []
    for i in range(0, len(bit_array), 8):
        byte_value = 0
        for j in range(8):
            byte_value = (byte_value << 1) | int(bit_array[i + j])
        bytes_list.append(byte_value)
    return bytes_list


def bytes_to_numpy_array(bytes_list):
    """Convert list of bytes to numpy array of bits"""
    bit_array = []
    for byte in bytes_list:
        for i in range(7, -1, -1):
            bit_array.append((byte >> i) & 1)
    return np.array(bit_array, dtype=np.uint8)


def sm4_encryption_binary(plaintext_input, key_input, rounds=32):
    """
    SM4 encryption with flexible input types

    Args:
        plaintext_input: Can be binary string, numpy array of bits, or list of bits
        key_input: Can be binary string or integer
        rounds: Number of rounds (default 32)

    Returns:
        numpy array of encrypted bits
    """
    # Handle plaintext input
    if isinstance(plaintext_input, str):
        # Binary string input
        plaintext_clean = plaintext_input.replace(" ", "")
        if len(plaintext_clean) != 128:
            raise ValueError(f"Plaintext must be 128 bits, got {len(plaintext_clean)}")
        P = binary_to_bytes(plaintext_clean)
    elif isinstance(plaintext_input, (np.ndarray, list)):
        # Numpy array or list of bits
        if len(plaintext_input) != 128:
            raise ValueError(f"Plaintext must be 128 bits, got {len(plaintext_input)}")
        P = numpy_array_to_bytes(plaintext_input)
    else:
        raise ValueError(f"Unsupported plaintext type: {type(plaintext_input)}")

    # Handle key input
    if isinstance(key_input, str):
        # Binary string input
        key_clean = key_input.replace(" ", "")
        if len(key_clean) != 128:
            raise ValueError(f"Key must be 128 bits, got {len(key_clean)}")
        MK = binary_to_bytes(key_clean)
    elif isinstance(key_input, int):
        # Integer input
        if key_input.bit_length() > 128:
            raise ValueError(f"Key integer too large for 128 bits")
        MK = integer_to_bytes(key_input)
    else:
        raise ValueError(f"Unsupported key type: {type(key_input)}")

    round_keys = key_expansion(MK)
    C = sm4_encrypt_block(P, round_keys, rounds)

    return bytes_to_numpy_array(C)


def sm4_decryption_binary(ciphertext_input, key_input, rounds=32):
    """
    SM4 decryption with flexible input types

    Args:
        ciphertext_input: Can be binary string, numpy array of bits, or list of bits
        key_input: Can be binary string or integer
        rounds: Number of rounds (default 32)

    Returns:
        numpy array of decrypted bits
    """
    # Handle ciphertext input
    if isinstance(ciphertext_input, str):
        # Binary string input
        ciphertext_clean = ciphertext_input.replace(" ", "")
        if len(ciphertext_clean) != 128:
            raise ValueError(
                f"Ciphertext must be 128 bits, got {len(ciphertext_clean)}"
            )
        C = binary_to_bytes(ciphertext_clean)
    elif isinstance(ciphertext_input, (np.ndarray, list)):
        # Numpy array or list of bits
        if len(ciphertext_input) != 128:
            raise ValueError(
                f"Ciphertext must be 128 bits, got {len(ciphertext_input)}"
            )
        C = numpy_array_to_bytes(ciphertext_input)
    else:
        raise ValueError(f"Unsupported ciphertext type: {type(ciphertext_input)}")

    # Handle key input
    if isinstance(key_input, str):
        # Binary string input
        key_clean = key_input.replace(" ", "")
        if len(key_clean) != 128:
            raise ValueError(f"Key must be 128 bits, got {len(key_clean)}")
        MK = binary_to_bytes(key_clean)
    elif isinstance(key_input, int):
        # Integer input
        if key_input.bit_length() > 128:
            raise ValueError(f"Key integer too large for 128 bits")
        MK = integer_to_bytes(key_input)
    else:
        raise ValueError(f"Unsupported key type: {type(key_input)}")

    round_keys = key_expansion(MK)
    D = sm4_decrypt_block(C, round_keys, rounds)

    return bytes_to_numpy_array(D)


def format_binary_output(binary_string, group_size=8):
    """Format binary string with spaces for readability"""
    return " ".join(
        binary_string[i : i + group_size]
        for i in range(0, len(binary_string), group_size)
    )
