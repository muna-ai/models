#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def scaled(x: float) -> float:
    """
    Scale a non-negative number, counting every call.
    """
    calls = 0
    try:
        if x < 0:
            raise ValueError("negative")
        result = x * 2.0
    except ValueError:
        result = 0.0
    finally:
        calls = calls + 1
    return result

if __name__ == "__main__":
    print(scaled(3.0))
    print(scaled(-1.0))
