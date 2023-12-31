def parse_target_size(value: str) -> tuple:
    return tuple(map(int, value.split(',')))
