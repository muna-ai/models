#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def list_comprehension_nested(count: int) -> list:
    """
    Test support for nested list comprehensions.
    """
    return [
        f"The numbers are {x} and {y}"
        for x in range(count)
        for y in range(count)
    ]