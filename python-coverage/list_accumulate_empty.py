#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

# `xs = []` is typed as `std::vector<$T>` with a fresh inference variable. The concat
# inside the loop binds `$T := std::string`, so the declaration finalizes to
# `std::vector<std::string>` and every iteration calls the plain `(vector, vector)`
# overload.

@compile()
def list_accumulate_empty(n: int) -> int:
    """
    Test loop-carried accumulation into an initially empty list.
    """
    xs = []
    for i in range(n):
        xs = xs + [str(i)]
    return len(xs)
