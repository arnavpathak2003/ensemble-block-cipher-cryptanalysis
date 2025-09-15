# Optimized HIGHT implementation with precomputed constants and lookup tables

import numpy as np

# Precomputed constants
DELTA = None
ROTATION_TABLES = None


def _initialize_constants():
    """Initialize precomputed constants once"""
    global DELTA, ROTATION_TABLES

    if DELTA is None:
        # Precompute delta constants
        s = [0, 1, 0, 1, 1, 0, 1]
        DELTA = [list_to_byte(s[::-1])]
        for i in range(1, 128):
            s.append(s[i + 2] ^ s[i - 1])
            DELTA.append(list_to_byte(s[i : i + 7][::-1]))

    if ROTATION_TABLES is None:
        # Precompute rotation lookup tables for f_0 and f_1
        ROTATION_TABLES = {"f0": [None] * 256, "f1": [None] * 256}

        for x in range(256):
            # f_0(x) = rotate_bits(x, 1) ^ rotate_bits(x, 2) ^ rotate_bits(x, 7)
            ROTATION_TABLES["f0"][x] = (
                rotate_bits(x, 1) ^ rotate_bits(x, 2) ^ rotate_bits(x, 7)
            )

            # f_1(x) = rotate_bits(x, 3) ^ rotate_bits(x, 4) ^ rotate_bits(x, 6)
            ROTATION_TABLES["f1"][x] = (
                rotate_bits(x, 3) ^ rotate_bits(x, 4) ^ rotate_bits(x, 6)
            )


def list_to_byte(lst):
    byte = 0
    for bit in lst:
        byte = (byte << 1) | bit
    return byte


def rotate_bits(x, n):
    return ((x << n) % 256) | (x >> (8 - n))


def whitening_key_generation(MK):
    # Precompute indices for whitening key generation
    WK = [None] * 8
    WK[0] = MK[12]
    WK[1] = MK[13]
    WK[2] = MK[14]
    WK[3] = MK[15]
    WK[4] = MK[0]
    WK[5] = MK[1]
    WK[6] = MK[2]
    WK[7] = MK[3]
    return WK


def subkey_generation_rounds(MK, rounds):
    """Generate subkeys for specific number of rounds with precomputed delta"""
    _initialize_constants()

    SK = [None] * (4 * rounds)
    delta_len = len(DELTA)

    for i in range(rounds):
        # Calculate which 16-round block we're in
        block_idx = i // 16
        round_in_block = i % 16

        # Precompute base indices
        sk_base = 4 * i
        delta_base = 16 * block_idx + round_in_block

        for j in range(4):
            sk_index = sk_base + j

            # For j=0,1 use MK[0:8], for j=2,3 use MK[8:16]
            if j < 2:
                mk_index = (j - round_in_block) % 8
                delta_index = delta_base + (j % 2) * 8
            else:
                mk_index = (j - round_in_block) % 8 + 8
                delta_index = delta_base + ((j - 2) % 2) * 8

            # Use modulo to handle bounds
            SK[sk_index] = (MK[mk_index] + DELTA[delta_index % delta_len]) % 256

    return SK


def encryption_key_schedule_rounds(MK, rounds):
    WK = whitening_key_generation(MK)
    SK = subkey_generation_rounds(MK, rounds)
    return WK, SK


def decryption_key_schedule_rounds(MK, rounds):
    WK = whitening_key_generation(MK)
    SK = subkey_generation_rounds(MK, rounds)[::-1]
    return WK, SK


def encryption_initial_transformation(P, WK):
    return [
        (P[0] + WK[0]) % 256,
        P[1],
        P[2] ^ WK[1],
        P[3],
        (P[4] + WK[2]) % 256,
        P[5],
        P[6] ^ WK[3],
        P[7],
    ]


def decryption_initial_transformation(C, WK):
    return [
        C[7],
        (C[0] - WK[4]) % 256,
        C[1],
        C[2] ^ WK[5],
        C[3],
        (C[4] - WK[6]) % 256,
        C[5],
        C[6] ^ WK[7],
    ]


def f_0(x):
    """Optimized f_0 using precomputed lookup table"""
    _initialize_constants()
    return ROTATION_TABLES["f0"][x]


def f_1(x):
    """Optimized f_1 using precomputed lookup table"""
    _initialize_constants()
    return ROTATION_TABLES["f1"][x]


def encryption_round_function(i, X_i, SK):
    sk_base = 4 * i
    return [
        X_i[7] ^ ((f_0(X_i[6]) + SK[sk_base + 3]) % 256),
        X_i[0],
        (X_i[1] + (f_1(X_i[0]) ^ SK[sk_base])) % 256,
        X_i[2],
        X_i[3] ^ ((f_0(X_i[2]) + SK[sk_base + 1]) % 256),
        X_i[4],
        (X_i[5] + (f_1(X_i[4]) ^ SK[sk_base + 2])) % 256,
        X_i[6],
    ]


def decryption_round_function(i, X_i, SK):
    sk_base = 4 * i
    return [
        X_i[1],
        (X_i[2] - (f_1(X_i[1]) ^ SK[sk_base + 3])) % 256,
        X_i[3],
        X_i[4] ^ ((f_0(X_i[3]) + SK[sk_base + 2]) % 256),
        X_i[5],
        (X_i[6] - (f_1(X_i[5]) ^ SK[sk_base + 1])) % 256,
        X_i[7],
        X_i[0] ^ ((f_0(X_i[7]) + SK[sk_base]) % 256),
    ]


def encryption_final_transformation(X_32, WK):
    return [
        (X_32[1] + WK[4]) % 256,
        X_32[2],
        X_32[3] ^ WK[5],
        X_32[4],
        (X_32[5] + WK[6]) % 256,
        X_32[6],
        X_32[7] ^ WK[7],
        X_32[0],
    ]


def decryption_final_transformation(X_32, WK):
    return [
        (X_32[0] - WK[0]) % 256,
        X_32[1],
        X_32[2] ^ WK[1],
        X_32[3],
        (X_32[4] - WK[2]) % 256,
        X_32[5],
        X_32[6] ^ WK[3],
        X_32[7],
    ]


def encryption_transformation_rounds(P, WK, SK, rounds):
    X_i = encryption_initial_transformation(P, WK)
    for i in range(rounds):
        X_i = encryption_round_function(i, X_i, SK)
    return encryption_final_transformation(X_i, WK)


def decryption_transformation_rounds(C, WK, SK, rounds):
    X_i = decryption_initial_transformation(C, WK)
    for i in range(rounds):
        X_i = decryption_round_function(i, X_i, SK)
    return decryption_final_transformation(X_i, WK)


# Binary conversion utilities
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


def hight_encryption_binary(plaintext_input, key_input, rounds=32):
    """
    HIGHT encryption with flexible input types

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
        if len(plaintext_clean) != 64:
            raise ValueError(f"Plaintext must be 64 bits, got {len(plaintext_clean)}")
        P = binary_to_bytes(plaintext_clean)
    elif isinstance(plaintext_input, (np.ndarray, list)):
        # Numpy array or list of bits
        if len(plaintext_input) != 64:
            raise ValueError(f"Plaintext must be 64 bits, got {len(plaintext_input)}")
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

    WK, SK = encryption_key_schedule_rounds(MK, rounds)
    C = encryption_transformation_rounds(P, WK, SK, rounds)

    return bytes_to_numpy_array(C)


def hight_decryption_binary(ciphertext_input, key_input, rounds=32):
    """
    HIGHT decryption with flexible input types

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
        if len(ciphertext_clean) != 64:
            raise ValueError(f"Ciphertext must be 64 bits, got {len(ciphertext_clean)}")
        C = binary_to_bytes(ciphertext_clean)
    elif isinstance(ciphertext_input, (np.ndarray, list)):
        # Numpy array or list of bits
        if len(ciphertext_input) != 64:
            raise ValueError(f"Ciphertext must be 64 bits, got {len(ciphertext_input)}")
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

    WK, SK = decryption_key_schedule_rounds(MK, rounds)
    D = decryption_transformation_rounds(C, WK, SK, rounds)

    return bytes_to_numpy_array(D)


def format_binary_output(binary_string, group_size=8):
    """Format binary string with spaces for readability"""
    return " ".join(
        binary_string[i : i + group_size]
        for i in range(0, len(binary_string), group_size)
    )
