# SPDX-License-Identifier: Apache-2.0

"""Functional tests for annotation classification notebook."""

import json
from pathlib import Path
import pytest

from tests.integration.notebook_utils import (
    execute_notebook_with_cell_injection,
    validate_notebook_execution,
)

pytestmark = pytest.mark.integration

ANNOTATION_NOTEBOOK_PATH = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "skills_tuning"
    / "instructlab"
    / "annotation_classification.ipynb"
)


def test_annotation_notebook_exists():
    """Verify annotation classification notebook exists."""
    assert ANNOTATION_NOTEBOOK_PATH.exists()


def test_annotation_notebook_dependencies():
    """Verify required flow and config files exist."""
    notebook_dir = ANNOTATION_NOTEBOOK_PATH.parent

    # Check flow files
    simple_flow = notebook_dir / "flows" / "simple_annotation.yaml"
    detailed_flow = notebook_dir / "flows" / "detailed_annotation.yaml"
    assert simple_flow.exists(), f"Simple flow file missing: {simple_flow}"
    assert detailed_flow.exists(), f"Detailed flow file missing: {detailed_flow}"

    # Check config files
    repo_root = Path(__file__).parent.parent.parent
    simple_config = (
        repo_root
        / "src"
        / "sdg_hub"
        / "configs"
        / "annotations"
        / "simple_annotations.yaml"
    )
    detailed_config = (
        repo_root
        / "src"
        / "sdg_hub"
        / "configs"
        / "annotations"
        / "detailed_annotations.yaml"
    )
    assert simple_config.exists(), f"Simple config file missing: {simple_config}"
    assert detailed_config.exists(), f"Detailed config file missing: {detailed_config}"


def _create_mock_setup_cell():
    """Create reusable mock setup cell for annotation classification testing."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["injected-mock"]},
        "outputs": [],
        "source": [
            "# Mock setup for annotation classification testing - MUST BE FIRST\n",
            "import sys\n",
            "from unittest.mock import MagicMock, patch\n",
            "\n",
            "# Mock categories that match guided_choice in flow configs\n",
            "expected_categories = ['World', 'Sports', 'Business', 'Sci/Tech']\n",
            "mock_responses = []\n",
            "for i in range(200):  # Enough for both simple and detailed flows\n",
            "    category = expected_categories[i % len(expected_categories)]\n",
            "    mock_response = MagicMock()\n",
            "    mock_response.choices = [MagicMock()]\n",
            "    mock_response.choices[0].message.content = category\n",
            "    mock_responses.append(mock_response)\n",
            "\n",
            "# Create a cycling iterator for responses\n",
            "response_iter = iter(mock_responses * 10)  # Repeat responses to ensure we don't run out\n",
            "\n",
            "# Mock the litellm completion function used by SDG Hub\n",
            "def mock_completion(*args, **kwargs):\n",
            "    return next(response_iter)\n",
            "\n",
            "# Patch the completion function in SDG Hub's client manager\n",
            "completion_patcher = patch('sdg_hub.blocks.llm.client_manager.completion', side_effect=mock_completion)\n",
            "completion_patcher.start()\n",
            "\n",
            "# Also mock OpenAI for the notebook's direct usage\n",
            "mock_client = MagicMock()\n",
            "mock_model = MagicMock()\n",
            "mock_model.id = 'meta-llama/Llama-3.3-70B-Instruct'\n",
            "mock_client.models.list.return_value.data = [mock_model]\n",
            "mock_client.chat.completions.create.side_effect = mock_responses\n",
            "\n",
            "openai_patcher = patch('openai.OpenAI', return_value=mock_client)\n",
            "openai_patcher.start()\n",
            "\n",
            "print('✅ Mock LLM setup complete for annotation classification')\n",
            "print(f'   - Mocked {len(mock_responses)} responses with categories: {expected_categories}')",
        ],
    }


def test_annotation_notebook_comprehensive(temp_output_dir: Path):
    """Comprehensive test of annotation classification notebook functionality.

    Tests:
    - End-to-end notebook execution without errors
    - Flow functionality and classification report generation
    - Data pipeline integrity from loading to evaluation
    """
    # Execute notebook with mock injection at position 1 (before imports)
    executed_notebook_path = execute_notebook_with_cell_injection(
        ANNOTATION_NOTEBOOK_PATH,
        injected_cells=[_create_mock_setup_cell()],
        parameters={},
        injection_position=1,  # Inject after first cell, before imports
        output_dir=temp_output_dir,
    )

    # Validate notebook executed successfully
    assert validate_notebook_execution(executed_notebook_path), (
        "Notebook execution failed with errors"
    )

    # Load executed notebook for comprehensive validation
    with open(executed_notebook_path, "r") as f:
        executed_notebook = json.load(f)

    # Initialize validation tracking
    executed_cells = 0
    total_code_cells = 0
    error_cells = []

    # Flow functionality evidence
    flows_loaded = False
    data_generated = False
    classification_reports_found = 0

    # Data pipeline evidence
    data_processing_evidence = {
        "dataset_loaded": False,
        "flows_created": False,
        "data_generated": False,
        "evaluation_performed": False,
    }

    # Comprehensive cell analysis
    for i, cell in enumerate(executed_notebook["cells"]):
        if cell.get("cell_type") == "code":
            total_code_cells += 1
            if cell.get("execution_count") is not None:
                executed_cells += 1

            # Check for errors in outputs
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    error_cells.append(
                        f"Cell {i}: {output.get('ename', 'Unknown error')}"
                    )

            # Analyze cell source for pipeline evidence
            cell_source = "".join(cell.get("source", []))

            # Flow functionality checks
            if "Flow.from_yaml" in cell_source:
                flows_loaded = True
                data_processing_evidence["flows_created"] = True
            if "flow.generate" in cell_source:
                data_generated = True
                data_processing_evidence["data_generated"] = True

            # Data pipeline checks
            if "load_dataset" in cell_source or "fancyzhx/ag_news" in cell_source:
                data_processing_evidence["dataset_loaded"] = True
            if "classification_report" in cell_source:
                data_processing_evidence["evaluation_performed"] = True

            # Check outputs for classification reports
            for output in cell.get("outputs", []):
                output_text = ""
                if output.get("output_type") == "stream":
                    output_text = "".join(output.get("text", []))
                elif output.get("output_type") in [
                    "execute_result",
                    "display_data",
                ]:
                    # Check both text/plain and text/html data
                    data = output.get("data", {})
                    if "text/plain" in data:
                        output_text = "".join(data["text/plain"])
                    elif "text/html" in data:
                        output_text = "".join(data["text/html"])

                if (
                    output_text
                    and "precision" in output_text
                    and "recall" in output_text
                    and "f1-score" in output_text
                ):
                    classification_reports_found += 1

    # Basic execution validation
    assert len(error_cells) == 0, f"Notebook has error cells: {error_cells}"
    assert executed_cells == total_code_cells, (
        f"Expected all {total_code_cells} code cells to execute, but only {executed_cells} executed"
    )

    # Flow functionality validation
    assert flows_loaded, "Flow loading code not found in notebook"
    assert data_generated, "Data generation code not found in notebook"
    assert classification_reports_found >= 1, (
        f"Expected at least 1 classification report, found {classification_reports_found}"
    )

    # Data pipeline integrity validation
    missing_steps = [
        step for step, found in data_processing_evidence.items() if not found
    ]
    assert len(missing_steps) == 0, f"Missing pipeline steps: {missing_steps}"
