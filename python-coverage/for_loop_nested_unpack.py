#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def for_loop_nested_unpack(count: int) -> int:
    """
    Test support for for-loops with nested tuple-unpacking targets.
    """
    pairs1 = [(i, i + 1) for i in range(count)]
    pairs2 = [(i * 2, i * 3) for i in range(count)]
    total = 0
    for (a, b), (c, d) in zip(pairs1, pairs2):
        total += a + b + c + d
    return total
