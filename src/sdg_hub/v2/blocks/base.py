"""
Base Block implementation for SDG Hub v2.

A Block represents a processing unit that operates on Huggingface Dataset objects.
"""

from typing import List, Optional, Union, Iterator, Any, Dict, Type, TypeVar
from pydantic import BaseModel, Field, ConfigDict
from datasets import Dataset

T = TypeVar('T', bound='Block')

class Block(BaseModel):
    """
    A processing block that operates on Huggingface Dataset objects.

    Attributes
    ----------
        name: Optional name for the block
        input_blocks: List of blocks that feed into this block
        output_blocks: List of blocks that this block feeds into
    """

    name: Optional[str] = Field(default=None, description="Optional name for the block")
    input_blocks: List["Block"] = Field(
        default_factory=list, description="List of input blocks"
    )
    output_blocks: List["Block"] = Field(
        default_factory=list, description="List of output blocks"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def connect(self, other: Union["Block", List["Block"]]) -> None:
        """
        Connect this block to one or more other blocks.

        Args:
            other: Either a single Block or a list of Blocks to connect to
        """
        if isinstance(other, list):
            for block in other:
                self._connect_single(block)
        else:
            self._connect_single(other)

    def _connect_single(self, other: "Block") -> None:
        """
        Connect this block to a single other block.

        Args:
            other: The block to connect to
        """
        if other not in self.output_blocks:
            self.output_blocks.append(other)
        if self not in other.input_blocks:
            other.input_blocks.append(self)

    def __rshift__(self, other: Union["Block", List["Block"]]) -> None:
        """
        Overload the >> operator to connect blocks.

        Args:
            other: Either a single Block or a list of Blocks to connect to
        """
        self.connect(other)

    def __rrshift__(self, other: List["Block"]) -> None:
        """
        Overload the >> operator when the list is on the left side.

        Args:
            other: List of Blocks to connect from
        """
        for block in other:
            block.connect(self)

    def run(self, *inputs: Dataset) -> Iterator[Any]:
        """
        Process the input datasets and yield output rows.

        This method should be implemented by subclasses.

        Args:
            *inputs: One or more Huggingface Dataset objects

        Yields
        ------
            Processed rows from the output dataset
        """
        raise NotImplementedError("Subclasses must implement the run method")

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """
        Custom serialization to handle circular references.
        Only serializes the name and references to connected blocks by name.
        """
        dump = {
            "name": self.name,
            "input_blocks": [block.name for block in self.input_blocks],
            "output_blocks": [block.name for block in self.output_blocks]
        }
        return dump

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any], block_registry: Dict[str, T]) -> T:
        """
        Reconstruct a block from its serialized form.
        
        Args:
            data: Serialized block data
            block_registry: Dictionary mapping block names to block instances
            
        Returns:
            Reconstructed block instance
            
        Raises:
            ValueError: If a referenced block is not found in the registry
        """
        # Create the block if it doesn't exist in the registry
        if data["name"] not in block_registry:
            block = cls(name=data["name"])
            block_registry[data["name"]] = block
        else:
            block = block_registry[data["name"]]
        
        # Connect to input blocks
        for input_name in data["input_blocks"]:
            if input_name not in block_registry:
                raise ValueError(f"Input block {input_name} not found in registry")
            block_registry[input_name] >> block
        
        # Connect to output blocks
        for output_name in data["output_blocks"]:
            if output_name not in block_registry:
                raise ValueError(f"Output block {output_name} not found in registry")
            block >> block_registry[output_name]
        
        return block

