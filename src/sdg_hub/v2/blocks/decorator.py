"""Block decorator for creating Block subclasses in SDG Hub v2."""

from typing import Callable, Type, TypeVar, Optional, Any, Iterator, Union
from datasets import Dataset
from .base import Block as BaseBlock
import uuid
import functools
import inspect

T = TypeVar("T", bound=BaseBlock)


def Block(
    name: Optional[str] = None, description: Optional[str] = None
) -> Union[Callable[[Callable[..., Iterator[Any]]], Type[BaseBlock]], Type[BaseBlock]]:
    """
    Decorator for creating Block subclasses.

    Args:
        name: Optional name for the block class
        description: Optional description for the block class

    Returns
    -------
        A decorator function that creates a Block subclass

    Example:
        @Block(name="MyBlock", description="A custom block")
        def my_block(inputs: Dataset) -> Iterator[Any]:
            for row in inputs:
                yield processed_row
    """
    # Handle case where @Block is used directly without arguments
    if callable(name) and description is None:
        # In this case, name is actually the function being decorated
        func = name
        name = None
        return _create_block_class(func, name=None, description=None)

    def decorator(func: Callable[..., Iterator[Any]]) -> Type[BaseBlock]:
        return _create_block_class(func, name, description)

    return decorator


def _create_block_class(
    func: Callable[..., Iterator[Any]], 
    name: Optional[str] = None, 
    description: Optional[str] = None
) -> Type[BaseBlock]:
    """Create a Block subclass from a function."""
    # Get the class name to use
    class_name = name or func.__name__
    doc_string = description or func.__doc__
    
    # Create the class dynamically with the name provided by the user
    attrs = {
        '__doc__': doc_string,
        '__module__': func.__module__,
        
        # Here we define the run method that uses the original function
        'run': lambda self, *inputs: func(*inputs),
        
        # Define the init method that handles unique names
        '__init__': lambda self, **kwargs: _block_init(self, class_name, kwargs),
    }
    
    # Create the class dynamically with the exact name the test expects
    BlockClass = type(class_name, (BaseBlock,), attrs)
    
    return BlockClass


def _block_init(self, class_name, kwargs):
    """Custom __init__ method for block classes."""
    # Generate a unique name if none is provided
    if 'name' not in kwargs:
        # Use class name with a unique identifier
        block_name = f"{class_name}_{str(uuid.uuid4())[:8]}"
        kwargs['name'] = block_name
    
    # Call the parent init
    super(self.__class__, self).__init__(**kwargs)
    
    # Auto-register with any active flow
    from ..flow import Flow
    current_flow = Flow.get_current()
    if current_flow:
        current_flow.add_block(self)
