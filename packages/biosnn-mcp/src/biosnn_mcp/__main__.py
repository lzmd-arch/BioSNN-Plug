import sys

from .server import main, main_write

if len(sys.argv) > 1 and sys.argv[1] == "write":
    raise SystemExit(main_write(sys.argv[2:]))

raise SystemExit(main(sys.argv[1:]))
