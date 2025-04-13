"""Tests for the Block decorator in SDG Hub v2."""

from datasets import Dataset
from sdg_hub.v2.blocks.base import Block as BaseBlock
from sdg_hub.v2.blocks.decorator import Block


def test_block_function():
    """Test block function that will be decorated."""

    def process_data(inputs: Dataset) -> Dataset:
        return inputs.map(lambda x: {"processed": True, **x})

    return process_data


def test_decorator_creation():
    """Test creating a block using the decorator."""

    # Create a block using the decorator
    @Block(name="TestBlock", description="A test block created with decorator")
    def test_block(inputs: Dataset) -> Dataset:
        return inputs.map(lambda x: {"processed": True, **x})

    # Verify the block was created correctly
    assert test_block.__name__ == "TestBlock"
    assert test_block.__doc__ == "A test block created with decorator"
    assert issubclass(test_block, BaseBlock)


def test_decorator_processing():
    """Test the processing functionality of a decorated block."""

    # Create a block using the decorator
    @Block(name="TestBlock")
    def test_block(inputs: Dataset) -> Dataset:
        return inputs.map(lambda x: {"processed": True, **x})

    # Create a test dataset
    data = {"text": ["Hello", "World"]}
    dataset = Dataset.from_dict(data)

    # Create an instance and process the data
    block = test_block()
    result = block.run(dataset)

    # Verify the processing
    assert isinstance(result, Dataset)
    assert all(row["processed"] for row in result)


def test_decorator_default_name():
    """Test that the decorator uses the function name as default."""

    @Block()
    def my_custom_block(inputs: Dataset) -> Dataset:
        return inputs

    assert my_custom_block.__name__ == "my_custom_block"


def test_decorator_connections():
    """Test that decorated blocks can be connected."""

    @Block(name="Block1")
    def block1(inputs: Dataset) -> Dataset:
        return inputs

    @Block(name="Block2")
    def block2(inputs: Dataset) -> Dataset:
        return inputs

    # Create instances and connect them
    b1 = block1()
    b2 = block2()
    b1.connect(b2)

    assert b1.output_blocks == [b2]
    assert b2.input_blocks == [b1]
