# Copyright (c) 2026 Nicholas Blauch. All rights reserved.
# This file is part of the original FOVI repository, used under the MIT License.

"""
Fast augmentation modules for KNNConv.

This module contains fast image augmentation operations optimized for
foveated vision processing.
"""

from .transforms import *
from .functional import *
from .functional_tensor import *
try:
    from .loader import *
except:
    # non-ffcv ops only
    pass