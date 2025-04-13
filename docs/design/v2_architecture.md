# SDG Hub v2 Architecture Design

## Table of Contents
1. [Version History and Migration](#version-history-and-migration)
2. [Architecture Overview](#architecture-overview)
3. [Block System Design](#block-system-design)
4. [API Documentation](#api-documentation)

## Version History and Migration

### Motivation for v2
SDG Hub v2 represents a significant redesign of the system, focusing on:
- Making block creation more intuitive and requiring less boilerplate code
- Consistent patterns for block creation and connection
- Rethinking of Block Connections and Flows leading to more extensible and powerful data generation workflows
- Designing components with testability in mind

### Key Changes from v1

- **Block Implementation**:
  - v1: 
  - v2: 
- **Connection Model**:
  - v1: Implicit connections via pipeline steps
  - v2: Explicit connections via dedicated methods and operator overloading
- **Metadata Handling**:
  - v1: Manual validation and handling of block metadata and configurations
  - v2: Pydantic Fields with automatic validation and serialization
- **Data Flow**:
  - v1: YAML-based flow definitions with dictionary-based data passing
  - v2: Python-based flow composition with direct connections and iterator-based data passing
- **Block Registration**:
  - v1: Explicit registration to a global registry
  - v2: Implicit class creation with proper naming and metadata
- **Configuration**:
  - v1: YAML files with manual loading and validation
  - v2: Python objects with type safety and validation
- **Code Organization**:
  - v1: Flat structure with monolithic files
  - v2: Hierarchical package organization with focused modules

### Migration Guide

## Architecture Overview

### Core Components
- **Block**: The fundamental processing unit
- **Dataset**: Input/output data container
- **Flow**: Orchestration of multiple Blocks
- **Decorators**: API enhancement tools

### Design Principles
1. **Composability**: Blocks can be easily combined and reused
2. **Extensibility**: New functionality can be added without modifying core code
3. **Maintainability**: Clear separation of concerns and well-defined interfaces

## Block System Design

### Block Concept
A Block is a processing unit that:
- Takes one or more Datasets as input
- Produces an Iterator of processed data
- Can be connected to other Blocks
- Is serializable and configurable

### Decorator Pattern Implementation
The @Block decorator provides a clean API for creating Block subclasses:

```python
@Block(name="MyBlock", description="Processes data")
def my_block(inputs: Dataset) -> Iterator[Any]:
    for row in inputs:
        yield process_row(row)
```

Key features:
- Automatic subclass creation
- Proper metaclass handling for Pydantic models
- Type preservation
- Documentation support

### Pydantic Integration
Blocks are Pydantic models, providing:
- Automatic validation
- Serialization/deserialization
- Configuration management
- Type checking

#### Serialization Support
Connections are preserved during serialization using block names:
```python
# Serialize block with connections
block_data = source_block.model_dump()
# Returns: {
#     "name": "Source",
#     "input_blocks": ["input_block_name"],
#     "output_blocks": ["processor_block_name"]
# }

# Reconstruct block with connections
reconstructed_block = Block.from_dict(block_data, block_registry)
```

### Connecting Blocks

Blocks can be connected to form processing pipelines using several methods:

1. **Direct Connection**
```python
@Block(name="Source")
def source(inputs: Dataset) -> Iterator[Any]:
    for row in inputs:
        yield {"processed": True, **row}

@Block(name="Processor")
def processor(inputs: Dataset) -> Iterator[Any]:
    for row in inputs:
        yield {"transformed": True, **row}

# Create instances
source_block = source()
processor_block = processor()

# Connect blocks directly
source_block.connect(processor_block)
```

2. **Operator Overloading**
```python
# Using the >> operator for more intuitive syntax
source_block >> processor_block

# Multiple connections in sequence
source_block >> processor_block >> output_block

# Connect to multiple blocks at once
source_block >> [processor1, processor2, processor3]

# Connect from multiple blocks
[source1, source2, source3] >> processor_block
```

#### Best Practices for Connections
1. **Clear Naming**: Use descriptive names for blocks to make connections clear
2. **Minimal Connections**: Keep the number of connections between blocks minimal
3. **Logical Grouping**: Group related blocks together
4. **Error Handling**: Handle connection errors gracefully
5. **Documentation**: Document complex connection patterns

#### Common Connection Patterns
1. **Linear Pipeline**
```python
@Block(name="Source")
def source(inputs: Dataset) -> Iterator[Any]:
    for row in inputs:
        yield row

@Block(name="Processor")
def processor(inputs: Dataset) -> Iterator[Any]:
    for row in inputs:
        yield {"processed": True, **row}

@Block(name="Output")
def output(inputs: Dataset) -> Iterator[Any]:
    for row in inputs:
        yield {"final": True, **row}

# Create and connect
source_block = source()
processor_block = processor()
output_block = output()

source_block >> processor_block >> output_block
```

2. **Fan-out Pattern**
```python
@Block(name="Source")
def source(inputs: Dataset) -> Iterator[Any]:
    for row in inputs:
        yield row

@Block(name="Processor1")
def processor1(inputs: Dataset) -> Iterator[Any]:
    for row in inputs:
        yield {"processed1": True, **row}

@Block(name="Processor2")
def processor2(inputs: Dataset) -> Iterator[Any]:
    for row in inputs:
        yield {"processed2": True, **row}

@Block(name="Aggregator")
def aggregator(inputs: Dataset) -> Iterator[Any]:
    for row in inputs:
        yield {"aggregated": True, **row}

# Create and connect
source_block = source()
processor1_block = processor1()
processor2_block = processor2()
aggregator_block = aggregator()

source_block >> [processor1_block, processor2_block] >> aggregator_block
```

3. **Fan-in Pattern**
```python
@Block(name="Source1")
def source1(inputs: Dataset) -> Iterator[Any]:
    for row in inputs:
        yield {"source1": True, **row}

@Block(name="Source2")
def source2(inputs: Dataset) -> Iterator[Any]:
    for row in inputs:
        yield {"source2": True, **row}

@Block(name="Processor")
def processor(inputs: Dataset) -> Iterator[Any]:
    for row in inputs:
        yield {"processed": True, **row}

# Create and connect
source1_block = source1()
source2_block = source2()
processor_block = processor()

[source1_block, source2_block] >> processor_block
```

## API Documentation

### Basic Usage
```python
from sdg_hub.v2.blocks import Block
from datasets import Dataset

@Block(name="Processor")
def process_data(inputs: Dataset) -> Dataset:
    return inputs.map(lambda x: {"processed": True, **x})
```

### Advanced Features
1. **Custom Names and Descriptions**
```python
@Block(
    name="CustomBlock",
    description="A block with custom configuration"
)
def custom_block(inputs: Dataset) -> Dataset:
    return inputs
```

2. **Block Connections**
```python
@Block(name="Source")
def source(inputs: Dataset) -> Dataset:
    return inputs

@Block(name="Processor")
def processor(inputs: Dataset) -> Dataset:
    return inputs

# Connect blocks
source_block = source()
processor_block = processor()
source_block.connect(processor_block)
```

3. **Type Safety**
```python
from typing import Iterator, Any

@Block()
def typed_block(inputs: Dataset) -> Iterator[Any]:
    for row in inputs:
        yield process_row(row)
```

### Best Practices
1. Always provide type hints
2. Use descriptive names and documentation
3. Keep blocks focused on a single responsibility
4. Handle errors gracefully
5. Use the connection API for complex workflows

### Common Patterns
1. **Data Transformation**
```python
@Block()
def transform(inputs: Dataset) -> Dataset:
    return inputs.map(transform_function)
```

2. **Filtering**
```python
@Block()
def filter_data(inputs: Dataset) -> Dataset:
    return inputs.filter(filter_function)
```

3. **Aggregation**
```python
@Block()
def aggregate(inputs: Dataset) -> Dataset:
    return inputs.reduce(aggregate_function)
```

### Limitations and Considerations
1. Blocks must be pure functions (no side effects)
2. Large datasets should be processed incrementally
3. Complex transformations may require multiple blocks
4. Memory usage should be monitored for large operations

## Future Enhancements