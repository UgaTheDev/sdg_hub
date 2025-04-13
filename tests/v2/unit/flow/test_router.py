"""Tests for the Router class in SDG Hub v2."""

import pytest
from datasets import Dataset
from sdg_hub.v2.flow import Router
from sdg_hub.v2.blocks import Block


@Block(name="TargetBlock1")
def target_block1(inputs: Dataset) -> Dataset:
    """Target block 1."""
    for row in inputs:
        yield {"id": row["id"], "target": 1, "value": row["value"]}


@Block(name="TargetBlock2")
def target_block2(inputs: Dataset) -> Dataset:
    """Target block 2."""
    for row in inputs:
        yield {"id": row["id"], "target": 2, "value": row["value"]}


@Block(name="DefaultBlock")
def default_block(inputs: Dataset) -> Dataset:
    """Default block."""
    for row in inputs:
        yield {"id": row["id"], "target": "default", "value": row["value"]}


@Block(name="SourceBlock")
def source_block(inputs: Dataset) -> Dataset:
    """Source block that generates data."""
    yield {"id": 1, "value": "A", "category": "cat1"}
    yield {"id": 2, "value": "B", "category": "cat2"}
    yield {"id": 3, "value": "C", "category": "cat1"}
    yield {"id": 4, "value": "D", "category": "cat3"}
    yield {"id": 5, "value": "E", "category": "cat2"}


def test_router_creation():
    """Test creating a router."""
    router = Router(name="test-router", description="Test router")
    assert router.name == "test-router"
    assert router.description == "Test router"
    assert router.routes == {}
    assert router.default_route is None


def test_router_routes():
    """Test adding routes to a router."""
    router = Router(name="test-router")
    
    target1 = target_block1()
    target2 = target_block2()
    default = default_block()
    
    # Add routes
    router.add_route(lambda row: row["category"] == "cat1", target1)
    router.add_route(lambda row: row["category"] == "cat2", target2)
    router.set_default(default)
    
    # Check routes
    assert len(router.routes) == 2
    assert router.default_route == default
    
    # Test route selection
    assert router.get_route({"category": "cat1"}) == target1
    assert router.get_route({"category": "cat2"}) == target2
    assert router.get_route({"category": "cat3"}) == default
    assert router.get_route({"category": "unknown"}) == default


def test_router_blocks():
    """Test creating routing blocks."""
    router = Router(name="test-router")
    
    target1 = target_block1()
    target2 = target_block2()
    default = default_block()
    source = source_block()
    
    # Add routes
    router.add_route(lambda row: row["category"] == "cat1", target1)
    router.add_route(lambda row: row["category"] == "cat2", target2)
    router.set_default(default)
    
    # Create routing blocks
    router_block, output_blocks = router.create_routing_blocks()
    
    # Check blocks
    assert "test-router_router" in router_block.name
    assert len(output_blocks) == 3
    
    # Connect blocks
    source >> router_block
    for output_block in output_blocks:
        if "route_to_TargetBlock1" in output_block.name:
            output_block >> target1
        elif "route_to_TargetBlock2" in output_block.name:
            output_block >> target2
        else:
            output_block >> default
    
    # Run the blocks (this is just to test that they run, not the actual routing logic)
    source_data = list(source.run(Dataset.from_dict({})))
    assert len(source_data) == 5
    
    router_data = list(router_block.run(Dataset.from_dict({
        "id": [row["id"] for row in source_data],
        "value": [row["value"] for row in source_data],
        "category": [row["category"] for row in source_data],
    })))
    
    assert len(router_data) == 1
    assert "router" in router_data[0]
    assert router_data[0]["router"] == "test-router"


def test_router_operator():
    """Test using the >> operator with a router."""
    router = Router(name="test-router")
    
    target = target_block1()
    
    # Connect with operator
    router >> target
    
    assert router.default_route == target
    
    # Test with a list
    router = Router(name="test-router")
    targets = [target_block1(), target_block2()]
    
    router >> targets
    
    # The operator should connect to the last block in the list when given a list
    assert router.default_route == targets[-1] 