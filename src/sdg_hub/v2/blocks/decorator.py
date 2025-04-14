"""Block decorator for creating Block subclasses in SDG Hub v2."""

import inspect
import uuid
from typing import (
    Callable,
    Type,
    TypeVar,
    Optional,
    Any,
    Iterator,
    Union,
    AsyncIterator,
    Awaitable,
)
from .base import Block as BaseBlock


T = TypeVar("T", bound=BaseBlock)


def Block(
    name: Optional[str] = None, description: Optional[str] = None
) -> Union[
    Callable[
        [
            Callable[
                ...,
                Union[
                    Iterator[Any],
                    AsyncIterator[Any],
                    Awaitable[Iterator[Any]],
                    Awaitable[AsyncIterator[Any]],
                ],
            ]
        ],
        Type[BaseBlock],
    ],
    Type[BaseBlock],
]:
    """
    Create Block subclasses using this decorator.

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

        # Or using async:
        @Block(name="AsyncBlock")
        async def async_block(inputs: Dataset) -> AsyncIterator[Any]:
            for row in inputs:
                yield row
    """
    # Handle case where @Block is used directly without arguments
    if callable(name) and description is None:
        # In this case, name is actually the function being decorated
        func = name
        name = None
        return _create_block_class(func, name=None, description=None)

    def decorator(
        func: Callable[
            ...,
            Union[
                Iterator[Any],
                AsyncIterator[Any],
                Awaitable[Iterator[Any]],
                Awaitable[AsyncIterator[Any]],
            ],
        ],
    ) -> Type[BaseBlock]:
        return _create_block_class(func, name, description)

    return decorator


def _create_block_class(
    func: Callable[
        ...,
        Union[
            Iterator[Any],
            AsyncIterator[Any],
            Awaitable[Iterator[Any]],
            Awaitable[AsyncIterator[Any]],
        ],
    ],
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Type[BaseBlock]:
    """Create a Block subclass from a function."""
    # Get the class name to use
    class_name = name or func.__name__
    doc_string = description or func.__doc__

    # Check if the function is async
    is_async = inspect.iscoroutinefunction(func)
    is_async_gen = inspect.isasyncgenfunction(func)

    # Define run and _run_sync methods based on the function type
    if is_async_gen:
        # For async generator functions, implement run directly
        async def async_run(self, *inputs):
            async for item in func(*inputs):
                yield item

        run_method = async_run
        run_sync_method = None
    elif is_async:
        # For regular async functions (not generators), implement run with async for
        async def async_run(self, *inputs):
            result = await func(*inputs)
            for item in result:
                yield item

        run_method = async_run
        run_sync_method = None
    else:
        # For sync functions, implement _run_sync and delegate to run_sync in run
        def sync_run(self, *inputs):
            return func(*inputs)

        async def async_run(self, *inputs):
            async for item in self.run_sync(*inputs):
                yield item

        run_method = async_run
        run_sync_method = sync_run

    # Create the class dynamically with the name provided by the user
    attrs = {
        "__doc__": doc_string,
        "__module__": func.__module__,
        # The async run method
        "run": run_method,
        # Define the init method that handles unique names
        "__init__": lambda self, **kwargs: _block_init(self, class_name, kwargs),
    }

    # Add _run_sync method if we have a sync function
    if run_sync_method:
        attrs["_run_sync"] = run_sync_method

    # Create the class dynamically with the exact name the test expects
    block_class = type(class_name, (BaseBlock,), attrs)

    return block_class


def _block_init(self, class_name, kwargs):
    """Initialize a block instance with a unique name and register it with the current flow.

    Args:
        class_name: The name of the block class
        kwargs: Keyword arguments for initializing the block
    """
    # Generate a unique name if none is provided
    if "name" not in kwargs:
        # Use class name with a unique identifier
        block_name = f"{class_name}_{str(uuid.uuid4())[:8]}"
        kwargs["name"] = block_name

    # Call the parent init
    super(self.__class__, self).__init__(**kwargs)

    # Auto-register with any active flow
    from ..flow import Flow

    current_flow = Flow.get_current()
    if current_flow:
        current_flow.add_block(self)
