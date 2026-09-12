#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile
from numpy import pi
from typing import Annotated

@compile()
def raise_exception() -> float:
    """
    Raise an exception.
    """
    raise RuntimeError(f"Cannot do things")

if __name__ == "__main__":
    print(raise_exception())