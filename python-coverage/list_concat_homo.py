#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile(
    tag="@yusuf/list-concat-homo",
    description="Test homogenous list concatenation."
)
def predict(name: str, age: int) -> list[str]:
    names = [name, name, name, name]
    more_names = [name, name]
    return names + more_names