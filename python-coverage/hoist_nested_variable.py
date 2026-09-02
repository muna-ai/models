#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# One major difference between Python and C++ is how each language handles scopes.
#
# In Python, there is no concept of scopes within a function. If you define a variable 
# within an indented block (e.g. `if` or `while` statement body), the variable remains 
# accessible outside that scope.
#
# But in C++, variables defined within a scope cannot be accessed outside that scope.
# So to emulate Python's behaviour in C++, we hoist all variables whose declaring block
# does not enclose every use, declaring them at function scope with their resolved type
# and value-initialized (e.g. `int64_t result{};`). Note that if the loop below never ran,
# Python would raise `UnboundLocalError` while the compiled function returns `0`.

from muna import compile
import numpy as np

@compile()
def hoist_nested_variable() -> np.int64:
    """
    Hoist a nested variable.
    """
    for i in range(2):
        result = 20
    return result