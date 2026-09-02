#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# Muna validates predictor inputs using Pydantic, so a client that omits an
# optional field still sees the field's default inside the predictor. The compiled
# predictor deserializes raw JSON, so the harness applies the defaults declared in the
# parameter's JSON schema before invoking the predictor. This model exercises a `None`
# default, a non-`None` default, and a nested model, all of which a client may omit.

from muna import compile
from pydantic import BaseModel

class Formatting(BaseModel):
    uppercase: bool = False
    suffix: str | None = None

class Greeting(BaseModel):
    name: str
    salutation: str = "Hello"
    formatting: Formatting = Formatting()

@compile()
def pydantic_defaults(request: Greeting) -> str:
    """
    Read Pydantic fields that a client may omit.
    """
    text = f"{request.salutation}, {request.name}"
    if request.formatting.uppercase:
        text = text.upper()
    if request.formatting.suffix:
        text = f"{text}{request.formatting.suffix}"
    return text
