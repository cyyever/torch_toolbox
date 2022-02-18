import os
import sys

module_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__))))
if module_dir not in sys.path:
    sys.path.append(module_dir)
