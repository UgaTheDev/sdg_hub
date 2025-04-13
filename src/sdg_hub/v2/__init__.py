"""
SDG Hub v2 package.
This is the new implementation of the SDG Hub package with improved design and features.
"""

from ._version import __version__
from .blocks import Block

# Core components will be imported here as they are implemented
# from .flow import Flow
# from .pipeline import Pipeline
# from .sdg import SDG

__all__ = [
    '__version__',
    'Block',
    # Add components to __all__ as they are implemented
] 