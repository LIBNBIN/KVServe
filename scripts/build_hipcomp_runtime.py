#!/usr/bin/env python3
"""Repository entry point; also available as python -m ...runtime.build_hipcomp."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kvserve_v1.compression.runtime.build_hipcomp import main

if __name__ == "__main__":
    main()
