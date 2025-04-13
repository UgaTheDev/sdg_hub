"""Block decorator for creating Block subclasses in SDG Hub v2."""

from typing import Callable, Type, TypeVar, Optional, Any, Iterator
from datasets import Dataset
from .base import Block as BaseBlock

T = TypeVar("T", bound=BaseBlock)


def Block(
    name: Optional[str] = None, description: Optional[str] = None
) -> Callable[[Callable[..., Iterator[Any]]], Type[BaseBlock]]:
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

    def decorator(func: Callable[..., Iterator[Any]]) -> Type[BaseBlock]:
        # Create a new subclass of BaseBlock
        class DecoratedBlock(BaseBlock):
            def run(self, *inputs: Dataset) -> Iterator[Any]:
                return func(*inputs)

        # Set class attributes
        class_name = name or func.__name__
        DecoratedBlock.__name__ = class_name
        DecoratedBlock.__qualname__ = class_name
        DecoratedBlock.__doc__ = description or func.__doc__
        DecoratedBlock.__module__ = func.__module__

        # Return the subclass
        return DecoratedBlock

    return decorator
