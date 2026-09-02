#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

# A `match` statement lowers to an `if`/`elif` chain: a value pattern compares with `==`,
# an or-pattern combines its alternatives with `or`, a guard is combined with `and`, and a
# capture or wildcard pattern is unconditional so it ends the chain.

@compile()
def match_(code: int) -> str:
    """
    Test support for `match` statements.
    """
    match code:
        case 200:                   return "ok"
        case 301 | 302:             return "redirect"
        case 404:                   return "not found"
        case value if value >= 500: return "server error"
        case _:                     return "unknown"
