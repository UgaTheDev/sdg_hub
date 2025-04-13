"""Tests for the Flow class in SDG Hub v2."""

import pytest
from datasets import Dataset
from sdg_hub.v2.flow import Flow
from sdg_hub.v2.blocks import Block


@Block(name="SourceBlock")
def source_block(inputs=None) -> Dataset:
    """Source block that generates data."""
    for i in range(5):
        yield {"id": i, "value": f"value-{i}"}


@Block(name="ProcessorBlock")
def processor_block(inputs=None) -> Dataset:
    """Processor block that transforms data."""
    if inputs:
        for row in inputs:
            yield {"id": row["id"], "processed_value": f"processed-{row['value']}"}
    else:
        for i in range(5):
            yield {"id": i, "processed_value": f"processed-default-{i}"}


@Block(name="SinkBlock")
def sink_block(inputs=None) -> Dataset:
    """Sink block that consumes data."""
    if inputs:
        for row in inputs:
            yield {"id": row["id"], "sink_value": f"sink-{row['processed_value']}"}
    else:
        for i in range(5):
            yield {"id": i, "sink_value": f"sink-default-{i}"}


def test_flow_creation():
    """Test flow creation and basic attributes."""
    flow = Flow(name="test-flow", description="Test flow")
    assert flow.name == "test-flow"
    assert flow.description == "Test flow"
    assert len(flow.blocks) == 0


def test_flow_context_manager():
    """Test flow context manager."""
    with Flow(name="test-flow") as flow:
        assert Flow.get_current() == flow
        
    assert Flow.get_current() is None


def test_flow_add_block():
    """Test adding blocks to a flow."""
    flow = Flow(name="test-flow")
    
    source = source_block()
    processor = processor_block()
    sink = sink_block()
    
    flow.add_block(source)
    flow.add_block(processor)
    flow.add_block(sink)
    
    assert len(flow.blocks) == 3
    assert flow.blocks[source.name] == source
    assert flow.blocks[processor.name] == processor
    assert flow.blocks[sink.name] == sink
    
    # Test adding a block with a duplicate name
    duplicate_source = source_block()
    duplicate_source.name = source.name  # Force duplicate name
    with pytest.raises(ValueError):
        flow.add_block(duplicate_source)


def test_flow_connections():
    """Test connecting blocks in a flow."""
    flow = Flow(name="test-flow")
    
    source = source_block()
    processor = processor_block()
    sink = sink_block()
    
    flow.add_block(source)
    flow.add_block(processor)
    flow.add_block(sink)
    
    # Connect the blocks
    source >> processor >> sink
    
    # Check connections
    assert processor in source.output_blocks
    assert source in processor.input_blocks
    assert sink in processor.output_blocks
    assert processor in sink.input_blocks


def test_flow_validation():
    """Test flow validation."""
    flow = Flow(name="test-flow")
    
    # Empty flow should be invalid
    assert flow.validate() is False
    
    source = source_block()
    processor = processor_block()
    sink = sink_block()
    
    flow.add_block(source)
    flow.add_block(processor)
    flow.add_block(sink)
    
    # Disconnected blocks should be invalid
    assert flow.validate() is False
    
    # Connect the blocks
    source >> processor >> sink
    
    # Connected blocks should be valid
    assert flow.validate() is True


def test_flow_execution():
    """Test flow execution."""
    flow = Flow(name="test-flow")
    
    source = source_block()
    processor = processor_block()
    sink = sink_block()
    
    flow.add_block(source)
    flow.add_block(processor)
    flow.add_block(sink)
    
    # Connect the blocks
    source >> processor >> sink
    
    # Execute the flow
    result = flow.run()
    
    # Check the result
    assert isinstance(result, Dataset)
    assert len(result) == 5
    assert "id" in result.column_names
    assert "sink_value" in result.column_names
    
    # Check specific values
    assert result[0]["id"] == 0
    assert result[0]["sink_value"] == "sink-processed-value-0"


def test_flow_serialization():
    """Test flow serialization and deserialization."""
    flow = Flow(name="test-flow", description="Test flow")
    
    source = source_block()
    processor = processor_block()
    sink = sink_block()
    
    flow.add_block(source)
    flow.add_block(processor)
    flow.add_block(sink)
    
    # Connect the blocks
    source >> processor >> sink
    
    # Save the flow to a temporary file
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        flow.save(tmp.name)
        
        # Load the flow from the file
        loaded_flow = Flow.load(tmp.name)
        
        # Clean up
        os.unlink(tmp.name)
    
    # Check the loaded flow
    assert loaded_flow.name == flow.name
    assert loaded_flow.description == flow.description
    assert len(loaded_flow.blocks) == len(flow.blocks)
    
    # Check that connections are preserved
    loaded_source = loaded_flow.blocks[source.name]
    loaded_processor = loaded_flow.blocks[processor.name]
    loaded_sink = loaded_flow.blocks[sink.name]
    
    assert loaded_processor in loaded_source.output_blocks
    assert loaded_source in loaded_processor.input_blocks
    assert loaded_sink in loaded_processor.output_blocks
    assert loaded_processor in loaded_sink.input_blocks
    
    # Execute the loaded flow
    result = loaded_flow.run()
    
    # Check the result
    assert isinstance(result, Dataset)
    assert len(result) == 5 