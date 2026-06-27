#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# Our compiler supports deleting variables eagerly, with the `del` operator.
# But two differences arise between Python and Muna:
#
# 1. In Python, `del` unbinds a name from a variable. This means that 
#    touching the name raises an `UnboundLocalError`. Muna does not 
#    preserve this behaviour; the name remains valid within its scope.
#
# 2. Muna deletes the underlying object, so depending on the compiled type,
#    you might see that the copy is uninitialized; or in the case below, 
#    the copy remains valid because assigning a list performs a value copy.

from muna import compile

@compile()
def delete_variable() -> int:
    """
    Test eager variable deletion.
    """
    numbers = [1, 2, 3, 4, 5]
    numbers_copy = numbers
    del numbers
    # using `numbers` here is an error in Python
    # but works when compiled by Muna
    return len(numbers_copy)

if __name__ == "__main__":
    print(delete_variable())