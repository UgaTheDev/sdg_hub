"""Integration tests for Block processing in SDG Hub v2."""

import pytest
from typing import Iterator, Any, AsyncIterator, Dict
from datasets import Dataset
from sdg_hub.v2.blocks import Block


class ProcessingBlock(Block):
    """A block that processes input data by adding a processed flag."""

    def _run_sync(self, *inputs: Dataset) -> Iterator[Any]:
        for dataset in inputs:
            for row in dataset:
                processed_row = dict(row)
                processed_row["processed"] = True
                yield processed_row
                
    async def run(self, *inputs: Dataset) -> AsyncIterator[Any]:
        async for item in self.run_sync(*inputs):
            yield item


class AsyncProcessingBlock(Block):
    """A block that processes input data asynchronously."""
    
    async def run(self, *inputs: Dataset) -> AsyncIterator[Dict[str, Any]]:
        for dataset in inputs:
            for row in dataset:
                processed_row = dict(row)
                processed_row["async_processed"] = True
                yield processed_row


class MultiInputBlock(Block):
    """A block that processes multiple input datasets."""

    def _run_sync(self, *inputs: Dataset) -> Iterator[Any]:
        combined_data = {}
        for dataset in inputs:
            for row in dataset:
                for key, value in row.items():
                    if key not in combined_data:
                        combined_data[key] = []
                    combined_data[key].append(value)
        yield combined_data
        
    async def run(self, *inputs: Dataset) -> AsyncIterator[Any]:
        async for item in self.run_sync(*inputs):
            yield item


@pytest.mark.asyncio
async def test_single_block_processing(sample_dataset):
    """Test processing with a single block."""
    block = ProcessingBlock(name="processor")
    results = []
    async for item in block.run(sample_dataset):
        results.append(item)

    assert len(results) == 2
    assert all("processed" in row for row in results)
    assert all(row["processed"] is True for row in results)
    assert results[0]["text"] == "hello"
    assert results[1]["text"] == "world"


@pytest.mark.asyncio
async def test_async_block_processing(sample_dataset):
    """Test processing with an async block."""
    block = AsyncProcessingBlock(name="async_processor")
    results = []
    async for item in block.run(sample_dataset):
        results.append(item)

    assert len(results) == 2
    assert all("async_processed" in row for row in results)
    assert all(row["async_processed"] is True for row in results)
    assert results[0]["text"] == "hello"
    assert results[1]["text"] == "world"


@pytest.mark.asyncio
async def test_connected_blocks_processing(sample_dataset):
    """Test processing with connected blocks."""
    # Create blocks
    block1 = ProcessingBlock(name="processor1")
    block2 = AsyncProcessingBlock(name="processor2")

    # Connect blocks
    block1 >> block2

    # Process data
    results1 = []
    async for item in block1.run(sample_dataset):
        results1.append(item)
        
    results2 = []
    async for item in block2.run(sample_dataset):
        results2.append(item)

    # Verify results
    assert len(results1) == 2
    assert len(results2) == 2
    assert all("processed" in row for row in results1)
    assert all("async_processed" in row for row in results2)


@pytest.mark.asyncio
async def test_multi_input_processing():
    """Test processing with multiple input datasets."""
    # Create test datasets
    dataset1 = Dataset.from_dict({"text": ["hello", "world"]})
    dataset2 = Dataset.from_dict({"number": [1, 2]})

    # Create and run block
    block = MultiInputBlock(name="combiner")
    results = []
    async for item in block.run(dataset1, dataset2):
        results.append(item)

    # Verify results
    assert len(results) == 1
    combined = results[0]
    assert "text" in combined
    assert "number" in combined
    assert combined["text"] == ["hello", "world"]
    assert combined["number"] == [1, 2]
