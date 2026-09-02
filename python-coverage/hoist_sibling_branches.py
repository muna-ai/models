#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# A Python local is function-scoped, so a name first bound inside one `if` branch can be
# freely rebound and read inside sibling branches. In C++ the first assignment would declare
# the variable inside that branch's block, leaving the sibling branches with an undeclared
# identifier. The emitter detects that the declaring block does not enclose every use and
# hoists a value-initialized declaration (`std::string label{};`) to function scope, even
# though `label` is never used outside the branches.

from muna import compile

@compile()
def hoist_sibling_branches(score: float) -> str:
    """
    Hoist a variable bound and consumed independently in sibling branches.
    """
    if score < 0.2:
        label = "low"
        return label.upper()
    elif score < 0.8:
        label = "medium"
        return label.upper()
    else:
        label = "high"
        return label.upper()
