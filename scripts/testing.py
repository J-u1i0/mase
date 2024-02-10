#! /usr/bin/env python3
# ---------------------------------------
# This script runs the hardware regression test
# ---------------------------------------
import sys 
import os
module_dir = os.path.join(os.path.dirname(__file__), '../machop/mase_components')
sys.path.append(module_dir)
from deps import MASE_HW_DEPS

print(MASE_HW_DEPS)
