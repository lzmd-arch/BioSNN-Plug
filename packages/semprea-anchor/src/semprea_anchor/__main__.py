"""让 `python -m semprea_anchor` 可执行。"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
