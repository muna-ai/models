#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile(
    tag="@muna/empty-list-append",
    description="Appending an item to an empty list"
)
def predict() -> list:
    result = []         # typed as `std::vector<$T>` with a fresh inference variable
    result.append(3)    # the `append(std::vector<T>&, const T&)` match binds `$T := int32_t`
    return result