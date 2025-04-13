"""
Blocks module for SDG Hub v2.
Contains the base Block class and various block implementations.
"""

# Import the original components
from .base import Block as _BaseBlock
from .decorator import Block as _DecoratorFunc

# Create a hybrid class that can be used both as a base class and a decorator
class Block(_BaseBlock):
    """
    Block class that also functions as a decorator.
    
    Can be used in two ways:
    1. As a base class: class MyBlock(Block)
    2. As a decorator: @Block(name="MyBlock")
    """
    
    @staticmethod
    def __call__(*args, **kwargs):
        """
        Make Block callable as a decorator.
        """
        # Get the original decorator function
        decorator = _DecoratorFunc(*args, **kwargs)
        
        # Return it so it can be used as @Block(...)
        return decorator

# Export the Block class
__all__ = ['Block']
