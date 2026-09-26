#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def check_bounds(
    value: int,
    min_value: int,
    max_value: int
) -> str:
    """
    Test compound if conditions with short-circuit evaluation.
    """
    if value >= min_value and value <= max_value:
        return "in_bounds"
    if value < min_value or value > max_value:
        return "out_of_bounds"
    return "unknown"
