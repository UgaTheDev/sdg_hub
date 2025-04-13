"""
Test configuration for SDG Hub v2 tests.
"""

import pytest
from datasets import Dataset

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
        def run(self, *inputs: Dataset):
            for dataset in inputs:
                for row in dataset:
                    yield dict(row, processed=True)
    
    return TestProcessingBlock(name="test_processor") 