#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def tuple_concat(name: str, age: int, score: float) -> str:
    """
    Test tuple concatenation.
    """
    user = (name, age)
    metrics = (score,)
    result = user + metrics
    return f"{result[0]}:{result[1]}:{result[2]}"