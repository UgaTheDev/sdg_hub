"""Examples of using Flows in SDG Hub v2."""

from typing import Optional
from datasets import Dataset
from sdg_hub.v2.flow import Flow
from sdg_hub.v2.blocks import Block


# Define some example blocks

@Block(name="LoadDataFromHub")
def load_data_from_hub(inputs=None, repo_id: Optional[str] = None, split: Optional[str] = None) -> Dataset:
    """Load a dataset from the Hub."""
    # In a real implementation, this would use the Hugging Face datasets library
    # For simplicity, we'll just return a sample dataset
    print(f"Loading data from repo: {repo_id}, split: {split}")
    
    # Generate sample data
    sample_data = [
        {"id": 1, "text": "This is a sample text", "label": "positive"},
        {"id": 2, "text": "Another example", "label": "negative"},
        {"id": 3, "text": "Yet another one", "label": "neutral"}
    ]
    
    for item in sample_data:
        yield item


class LLM:
    """Base LLM class for examples."""
    def __init__(self, model: str):
        self.model = model
        self.model_name = model
    
    def generate(self, prompt: str, **kwargs):
        """Generate text from the LLM."""
        return f"Generated text from {self.model}: {prompt}"


class OpenAILLM(LLM):
    """OpenAI LLM."""
    pass


class MistralLLM(LLM):
    """Mistral LLM."""
    pass


class VertexAILLM(LLM):
    """VertexAI LLM."""
    pass


@Block(name="TextGeneration")
def text_generation(inputs=None, llm: Optional[LLM] = None) -> Dataset:
    """Generate text using an LLM."""
    if not inputs:
        # Default behavior when no inputs provided
        yield {
            "id": 0,
            "text": "Default text",
            "generation": "Default generation",
            "model_name": llm.model_name if llm else "none"
        }
        return
    
    try:
        # Handle both Dataset objects and iterables
        if hasattr(inputs, '__iter__'):
            for row in inputs:
                if isinstance(row, dict) and "text" in row:
                    prompt = row["text"]
                    generation = llm.generate(prompt) if llm else f"No LLM provided: {prompt}"
                    yield {
                        "id": row.get("id", 0),
                        "text": prompt,
                        "generation": generation,
                        "model_name": llm.model_name if llm else "none"
                    }
                else:
                    # Handle the case where the row doesn't have the expected format
                    yield {
                        "id": 0,
                        "text": str(row),
                        "generation": "Couldn't process input",
                        "model_name": llm.model_name if llm else "none"
                    }
        else:
            # Handle single objects that are not iterable
            yield {
                "id": 0,
                "text": str(inputs),
                "generation": llm.generate(str(inputs)) if llm else f"No LLM provided: {inputs}",
                "model_name": llm.model_name if llm else "none"
            }
    except Exception as e:
        # Provide fallback behavior if an error occurs
        print(f"Error in text_generation: {e}")
        yield {
            "id": 0,
            "text": "Error occurred",
            "generation": f"Error: {str(e)}",
            "model_name": llm.model_name if llm else "none"
        }


@Block(name="GroupColumns")
def group_columns(inputs=None, columns: Optional[list] = None, output_columns: Optional[list] = None) -> Dataset:
    """Group multiple columns into lists."""
    if not inputs:
        yield {"id": 0, "grouped_data": "Default grouped data"}
        return
        
    if not columns or not output_columns:
        # Pass through if no columns specified
        for row in inputs:
            yield row
        return
        
    for row in inputs:
        try:
            output = {k: v for k, v in row.items() if isinstance(k, (str, int))}
            for col_idx, col in enumerate(columns):
                if col_idx < len(output_columns) and col in row:
                    # Group values from different rows with the same ID
                    output[output_columns[col_idx]] = [row[col]]
            yield output
        except (TypeError, KeyError) as e:
            # Handle errors and provide a default output if there's an issue
            print(f"Error processing row {row}: {e}")
            yield {"id": row.get("id", 0), "error": str(e)}


def example_basic_flow():
    """Basic flow example."""
    with Flow(name="basic-flow", description="Basic flow example") as flow:
        # Define blocks
        load_dataset = load_data_from_hub(name="load_dataset")
        
        combine_generations = group_columns(
            name="combine_generations",
            columns=["generation", "model_name"],
            output_columns=["generations", "model_names"],
        )
        
        # Define LLMs and tasks
        openai_llm = OpenAILLM(model="gpt-4-0125-preview")
        mistral_llm = MistralLLM(model="mistral-large-2402")
        vertex_llm = VertexAILLM(model="gemini-1.5-pro")
        
        openai_task = text_generation(name="text_generation_with_openai", llm=openai_llm)
        mistral_task = text_generation(name="text_generation_with_mistral", llm=mistral_llm)
        vertex_task = text_generation(name="text_generation_with_vertex", llm=vertex_llm)
        
        # Connect blocks
        load_dataset >> openai_task >> combine_generations
        load_dataset >> mistral_task >> combine_generations
        load_dataset >> vertex_task >> combine_generations
        
        # Print flow validation status to help debug
        print(f"Flow validation: {flow.validate()}")
        print(f"Flow blocks: {len(flow.blocks)}")
        for name, block in flow.blocks.items():
            print(f"  {name}: inputs={len(block.input_blocks)}, outputs={len(block.output_blocks)}")
    
    # Run the flow
    result = flow.run(
        parameters={
            "load_dataset": {
                "repo_id": "distilabel-internal-testing/instruction-dataset-mini",
                "split": "test",
            },
            "text_generation_with_openai": {
                "llm": {
                    "generation_kwargs": {
                        "temperature": 0.7,
                        "max_new_tokens": 512,
                    }
                }
            },
            "text_generation_with_mistral": {
                "llm": {
                    "generation_kwargs": {
                        "temperature": 0.7,
                        "max_new_tokens": 512,
                    }
                }
            },
            "text_generation_with_vertex": {
                "llm": {
                    "generation_kwargs": {
                        "temperature": 0.7,
                        "max_new_tokens": 512,
                    }
                }
            },
        },
    )
    
    print(f"Result: {result}")
    return flow, result


def example_flow_with_list():
    """Flow with list of tasks example."""
    with Flow(name="flow-with-list", description="Flow with list of tasks example") as flow:
        # Define blocks
        load_dataset = load_data_from_hub(name="load_dataset")
        
        combine_generations = group_columns(
            name="combine_generations",
            columns=["generation", "model_name"],
            output_columns=["generations", "model_names"],
        )
        
        # Define LLMs and create tasks in a loop
        for i, llm in enumerate([
            OpenAILLM(model="gpt-4-0125-preview"),
            MistralLLM(model="mistral-large-2402"),
            VertexAILLM(model="gemini-1.5-pro"),
        ]):
            task = text_generation(name=f"text_generation_with_model_{i}", llm=llm)
            load_dataset >> task >> combine_generations
        
        # Print flow validation status to help debug
        print(f"Flow with list validation: {flow.validate()}")
        print(f"Flow with list blocks: {len(flow.blocks)}")
        for name, block in flow.blocks.items():
            print(f"  {name}: inputs={len(block.input_blocks)}, outputs={len(block.output_blocks)}")
    
    # Run the flow
    result = flow.run(
        parameters={
            "load_dataset": {
                "repo_id": "distilabel-internal-testing/instruction-dataset-mini",
                "split": "test",
            }
        },
    )
    
    print(f"Result: {result}")
    return flow, result


def save_and_load_flow():
    """Example of saving and loading a flow."""
    # Create a flow
    flow, _ = example_basic_flow()
    
    # Save the flow
    flow.save("basic_flow.json")
    
    # Load the flow
    loaded_flow = Flow.load("basic_flow.json")
    
    print(f"Loaded flow: {loaded_flow.name}")
    print(f"Number of blocks: {len(loaded_flow.blocks)}")
    
    # Print blocks and connections in the loaded flow
    for name, block in loaded_flow.blocks.items():
        print(f"  {name}: inputs={len(block.input_blocks)}, outputs={len(block.output_blocks)}")
    
    # Validate the loaded flow
    print(f"Loaded flow validation: {loaded_flow.validate()}")
    
    return loaded_flow


if __name__ == "__main__":
    # Run examples
    example_basic_flow()
    example_flow_with_list()
    save_and_load_flow() 