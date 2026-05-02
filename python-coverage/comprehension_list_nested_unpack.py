#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def list_comprehension_nested_unpack(count: int) -> list:
    """
    Test support for list comprehensions with nested tuple-unpacking targets.
    """
    pairs1 = [(i, i + 1) for i in range(count)]
    pairs2 = [(i * 2, i * 3) for i in range(count)]
    return [
        f"{a}, {b}, {c}, {d}"
        for (a, b), (c, d) in zip(pairs1, pairs2)
    ]
