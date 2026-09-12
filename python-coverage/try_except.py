#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def parse_or_measure(x: str) -> int:
    """
    Parse an integer, falling back to the length of the error message.
    """
    try:
        y = int(x)
    except (ValueError, KeyError) as e:
        y = len(str(e))
    except Exception as err:
        raise RuntimeError("wrapped") from err
    except:
        raise
    return y

if __name__ == "__main__":
    print(parse_or_measure("12"))
    print(parse_or_measure("twelve"))
