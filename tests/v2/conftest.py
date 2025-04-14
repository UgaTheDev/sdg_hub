"""
Test configuration for SDG Hub v2 tests.
"""

import pytest
from datasets import Dataset
from typing import Iterator, AsyncIterator, Any, Dict

@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return Dataset.from_dict({
        "text": ["hello", "world"],
        "label": [0, 1]
    })

@pytest.fixture
def processing_block():
    """Create a sample processing block."""
    from sdg_hub.v2.blocks import Block
    
    class TestProcessingBlock(Block):
        def _run_sync(self, *inputs: Dataset) -> Iterator[Any]:
            for dataset in inputs:
                for row in dataset:
                    yield dict(row, processed=True)
        
        async def run(self, *inputs: Dataset) -> AsyncIterator[Dict]:
            async for item in self.run_sync(*inputs):
                yield item
    
    return TestProcessingBlock(name="test_processor")

@pytest.fixture
def async_processing_block():
    """Create a sample async processing block."""
    from sdg_hub.v2.blocks import Block
    
    class TestAsyncProcessingBlock(Block):
        async def run(self, *inputs: Dataset) -> AsyncIterator[Dict]:
            for dataset in inputs:
                for row in dataset:
                    yield dict(row, async_processed=True)
    
    return TestAsyncProcessingBlock(name="test_async_processor") 