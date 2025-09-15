import random


def generate_unique_large_integers(num_elements, num_bits, seed=32):
    """
    Generates a list of unique large integers within a specified bit range.

    Args:
        num_elements (int): The number of unique elements to generate.
        num_bits (int): The maximum number of bits for each integer.

    Returns:
        list: A list of unique large integers.
    """
    if num_elements > 2**num_bits:
        raise ValueError(
            f"Cannot generate {num_elements} unique numbers "
            f"with only {num_bits} bits (max {2**num_bits} unique values)."
        )

    random.seed(seed)
    generated_numbers = set()
    while len(generated_numbers) < num_elements:
        num = random.getrandbits(num_bits)
        generated_numbers.add(num)

    return list(generated_numbers)


def binary_str_to_list(binary_str):
    """
    Converts a binary string to a list of integers (0s and 1s).

    Args:
        binary_str (str): A string representing a binary number.

    Returns:
        list: A list of integers (0s and 1s).
    """
    return [int(bit) for bit in binary_str]
