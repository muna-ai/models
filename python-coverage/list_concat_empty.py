#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

# `names = []` is typed as `std::vector<$T>` with a fresh inference variable.
# Concatenating it with a `std::vector<std::string>` unifies `$T := std::string`,
# so the declaration finalizes to `std::vector<std::string>`.

@compile()
def list_concat_empty(name: str) -> list:
    """
    Test heterogenous list concatenation.
    """
    names = []
    names = names + [name, name]
    return len(names)