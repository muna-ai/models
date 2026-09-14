#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def generator(sentence: str) -> str:
    """
    Test compiling a generator expression.
    """
    parts = (c.upper() for c in sentence.split())
    return next(parts)