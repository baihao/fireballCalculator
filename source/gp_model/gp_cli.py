#!/usr/bin/env python3
"""
在 ``gp_model/`` 目录内可直接运行，无需 ``cd ..`` 或 ``PYTHONPATH``::

    cd /path/to/fireball_calculator/source/gp_model
    python gp_cli.py train --data-dir ..
"""
from __future__ import annotations

import sys
from pathlib import Path

# 包所在目录为 source/gp_model/，须把 source/ 加入路径才能 import gp_model
_SOURCE = Path(__file__).resolve().parent.parent
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from gp_model.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
