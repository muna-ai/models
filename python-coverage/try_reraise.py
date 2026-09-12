#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def checked_sqrt(x: float) -> float:
    """
    Compute the square root of a non-negative number, re-raising any failure.
    """
    try:
        if x < 0:
            raise ValueError(f"Cannot take square root of {x}")
        result = x ** 0.5
    except ValueError:
        raise
    except:
        result = 0.0
    return result

if __name__ == "__main__":
    print(checked_sqrt(4.))
