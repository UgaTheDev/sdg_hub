"""Unit tests for the Block base class in SDG Hub v2."""

from typing import Iterator, Any
from datasets import Dataset
from sdg_hub.v2.blocks import Block


class DummyBlock(Block):
    """A simple test block that passes through the input data."""

    def run(self, *inputs: Dataset) -> Iterator[Any]:
        for dataset in inputs:
            for row in dataset:
                yield row


def test_block_initialization():
    """Test block initialization and basic attributes."""
    block = DummyBlock(name="test_block")
    assert block.name == "test_block"
    assert len(block.input_blocks) == 0
    assert len(block.output_blocks) == 0


def test_block_connection_single():
    """Test single block connection functionality."""
    block1 = DummyBlock(name="block1")
    block2 = DummyBlock(name="block2")

    block1 >> block2

    assert block2 in block1.output_blocks
    assert block1 in block2.input_blocks
    assert len(block1.output_blocks) == 1
    assert len(block2.input_blocks) == 1


def test_block_connection_multiple():
    """Test multiple block connections."""
    block1 = DummyBlock(name="block1")
    block2 = DummyBlock(name="block2")
    block3 = DummyBlock(name="block3")

    block1 >> [block2, block3]

    assert block2 in block1.output_blocks
    assert block3 in block1.output_blocks
    assert block1 in block2.input_blocks
    assert block1 in block3.input_blocks
    assert len(block1.output_blocks) == 2
    assert len(block2.input_blocks) == 1
    assert len(block3.input_blocks) == 1


def test_block_connection_list_to_single():
    """Test connecting multiple blocks to a single block."""
    block1 = DummyBlock(name="block1")
    block2 = DummyBlock(name="block2")
    block3 = DummyBlock(name="block3")

    [block2, block3] >> block1

    assert block1 in block2.output_blocks
    assert block1 in block3.output_blocks
    assert block2 in block1.input_blocks
    assert block3 in block1.input_blocks
    assert len(block1.input_blocks) == 2
    assert len(block2.output_blocks) == 1
    assert len(block3.output_blocks) == 1


def test_block_serialization():
    """Test block serialization with pydantic."""
    block1 = DummyBlock(name="serializable")
    block2 = DummyBlock(name="connected")
    block1 >> block2

    block_dict = block1.model_dump()

    assert block_dict["name"] == "serializable"
    assert block_dict["output_blocks"] == ["connected"]
    assert block_dict["input_blocks"] == []


def test_block_reconstruction():
    """Test block reconstruction from serialized form."""
    # Create initial blocks and connections
    block1 = DummyBlock(name="block1")
    block2 = DummyBlock(name="block2")
    block3 = DummyBlock(name="block3")
    block1 >> [block2, block3]
    [block2, block3] >> block1

    # Serialize blocks
    block1_dict = block1.model_dump()
    block2_dict = block2.model_dump()
    block3_dict = block3.model_dump()

    # Create new registry and reconstruct blocks
    registry = {}
    
    # First create all blocks without connections
    for block_dict in [block1_dict, block2_dict, block3_dict]:
        if block_dict["name"] not in registry:
            block = DummyBlock(name=block_dict["name"])
            registry[block_dict["name"]] = block
    
    # Then reconstruct connections
    for block_dict in [block1_dict, block2_dict, block3_dict]:
        DummyBlock.from_dict(block_dict, registry)

    # Get reconstructed blocks
    reconstructed1 = registry["block1"]
    reconstructed2 = registry["block2"]
    reconstructed3 = registry["block3"]

    # Verify reconstructions
    assert reconstructed1.name == "block1"
    assert reconstructed2.name == "block2"
    assert reconstructed3.name == "block3"

    # Verify connections
    assert reconstructed2 in reconstructed1.output_blocks
    assert reconstructed3 in reconstructed1.output_blocks
    assert reconstructed1 in reconstructed2.input_blocks
    assert reconstructed1 in reconstructed3.input_blocks
    assert reconstructed2 in reconstructed1.input_blocks
    assert reconstructed3 in reconstructed1.input_blocks
    assert reconstructed1 in reconstructed2.output_blocks
    assert reconstructed1 in reconstructed3.output_blocks


def test_block_reconstruction_missing_block():
    """Test block reconstruction with missing referenced block."""
    block_dict = {
        "name": "block1",
        "input_blocks": ["missing_block"],
        "output_blocks": []
    }
    registry = {}

    # Attempt reconstruction should raise ValueError
    import pytest
    with pytest.raises(ValueError, match="Input block missing_block not found in registry"):
        DummyBlock.from_dict(block_dict, registry)
