"""Entry point for `python -m wien2k_gen`."""
import sys

from wien2k_gen.cli import main

if __name__ == "__main__":
    sys.exit(main())
