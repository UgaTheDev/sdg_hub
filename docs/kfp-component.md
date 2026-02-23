# KFP Component

SDG Hub includes a Kubeflow Pipelines (KFP) component for running flows inside pipeline steps.

## Install

Install the optional KFP dependency set:

```bash
# With pip
pip install sdg-hub[kfp]

# With uv
uv pip install sdg-hub[kfp]
```

## Local Workflow

The repository includes a reference KFP workflow under `kfp/`:

```bash
# Build local component image
podman build -t sdg-hub-kfp:dev -f kfp/Dockerfile .

# Run KFP component tests
uv run python -m pytest tests/kfp/ -v

# Compile sample pipeline definition
uv run python kfp/pipeline.py
```

## Pipeline Usage

```python
from sdg_hub.kfp import sdg

from kfp import dsl


@dsl.pipeline(name="sdg-hub-pipeline")
def sdg_pipeline():
    sdg(
        input_pvc_path="/mnt/data/input.jsonl",
        flow_id="your-flow-id",
        model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
    )
```

## Related Files

- `kfp/README.md`: setup and local testing guide
- `kfp/ARCHITECTURE.md`: design decisions and interface details
- `kfp/pipeline.py`: sample pipeline composition
- `src/sdg_hub/kfp/component.py`: component implementation
