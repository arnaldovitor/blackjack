def string_to_tuple(value: str) -> tuple:
    """Converts a comma-separated string of integers into a tuple of integers.

    Args:
        value (str): A string containing comma-separated integers.

    Returns:
        tuple: A tuple of integers parsed from the input string.
    """
    return tuple(map(int, value.split(',')))
