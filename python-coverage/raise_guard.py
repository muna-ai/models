#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def checked_sqrt(x: float) -> float:
    """
    Compute the square root of a non-negative number.
    """
    if x < 0:
        raise ValueError(f"Cannot take square root of {x}")
    return x ** 0.5

if __name__ == "__main__":
    print(checked_sqrt(4.))
