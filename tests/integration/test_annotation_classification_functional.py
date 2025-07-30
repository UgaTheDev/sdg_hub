# SPDX-License-Identifier: Apache-2.0

"""Functional tests for annotation classification notebook."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from tests.integration.notebook_utils import (
    execute_notebook_with_params,
    execute_notebook_with_cell_injection,
    validate_notebook_execution,
    extract_notebook_outputs,
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


def test_annotation_notebook_execution(temp_output_dir: Path):
    """Test end-to-end execution of annotation classification notebook with mocked LLM."""
    # Create mock setup cell specific to annotation classification
    mock_setup_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["injected-mock"]},
        "outputs": [],
        "source": [
            "# Mock setup for annotation classification testing\n",
            "from unittest.mock import MagicMock\n",
            "import openai\n",
            "\n",
            "# Mock categories that match guided_choice in flow configs\n",
            "expected_categories = ['World', 'Sports', 'Business', 'Sci/Tech']\n",
            "mock_responses = []\n",
            "for i in range(200):  # Enough for both simple and detailed flows\n",
            "    category = expected_categories[i % len(expected_categories)]\n",
            "    mock_responses.append(\n",
            "        MagicMock(choices=[MagicMock(message=MagicMock(content=category))])\n",
            "    )\n",
            "\n",
            "# Create mock client\n",
            "mock_client = MagicMock()\n",
            "mock_model = MagicMock()\n",
            "mock_model.id = 'meta-llama/Llama-3.3-70B-Instruct'\n",
            "mock_client.models.list.return_value.data = [mock_model]\n",
            "mock_client.chat.completions.create.side_effect = mock_responses\n",
            "\n",
            "# Replace OpenAI constructor to return our mock client\n",
            "openai.OpenAI = lambda *args, **kwargs: mock_client\n",
            "print('✅ Mock LLM setup complete for annotation classification')"
        ]
    }
    
    # Execute notebook with mock injection
    executed_notebook_path = execute_notebook_with_cell_injection(
        ANNOTATION_NOTEBOOK_PATH,
        injected_cells=[mock_setup_cell],
        parameters={},
        output_dir=temp_output_dir,
    )

    # Validate notebook executed successfully
    assert validate_notebook_execution(executed_notebook_path), (
        "Notebook execution failed with errors"
    )

    # Load executed notebook for detailed validation
    with open(executed_notebook_path, "r") as f:
        executed_notebook = json.load(f)

    # Verify all code cells executed
    executed_cells = 0
    total_code_cells = 0
    error_cells = []

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

    assert len(error_cells) == 0, f"Notebook has error cells: {error_cells}"
    assert executed_cells == total_code_cells, (
        f"Expected all {total_code_cells} code cells to execute, but only {executed_cells} executed"
    )


def test_annotation_flows_functional(temp_output_dir: Path):
    """Test that both annotation flows work correctly with real data processing."""

    # Create enough mock responses for the full notebook execution
    expected_categories = ["World", "Sports", "Business", "Sci/Tech"]
    mock_responses = []
    for i in range(200):  # Enough for both flows
        category = expected_categories[i % len(expected_categories)]
        mock_responses.append(
            MagicMock(choices=[MagicMock(message=MagicMock(content=category))])
        )

    with patch("openai.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.id = "meta-llama/Llama-3.3-70B-Instruct"
        mock_client.models.list.return_value.data = [mock_model]
        mock_client.chat.completions.create.side_effect = mock_responses
        mock_openai_class.return_value = mock_client

        executed_notebook_path = execute_notebook_with_params(
            ANNOTATION_NOTEBOOK_PATH,
            {},  # No parameters
            output_dir=temp_output_dir,
        )

        # Validate flows executed correctly
        with open(executed_notebook_path, "r") as f:
            executed_notebook = json.load(f)

        # Look for evidence of successful flow execution
        flows_loaded = False
        data_generated = False
        classification_reports_found = 0

        for cell in executed_notebook["cells"]:
            if cell.get("cell_type") == "code":
                # Check source code for flow operations
                cell_source = "".join(cell.get("source", []))
                if "Flow.from_yaml" in cell_source:
                    flows_loaded = True
                if "flow.generate" in cell_source:
                    data_generated = True

                # Check outputs for classification reports (any output type)
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

        assert flows_loaded, "Flow loading code not found in notebook"
        assert data_generated, "Data generation code not found in notebook"
        assert classification_reports_found >= 1, (
            f"Expected at least 1 classification report, found {classification_reports_found}"
        )


def test_annotation_data_pipeline_integrity(temp_output_dir: Path):
    """Test that data flows correctly through the annotation pipeline blocks."""

    # Mock responses that match expected categories
    expected_categories = ["World", "Sports", "Business", "Sci/Tech"]
    mock_responses = []

    # Generate consistent mock responses for full notebook
    for i in range(200):
        category = expected_categories[i % len(expected_categories)]
        mock_responses.append(
            MagicMock(choices=[MagicMock(message=MagicMock(content=category))])
        )

    with patch("openai.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.id = "meta-llama/Llama-3.3-70B-Instruct"
        mock_client.models.list.return_value.data = [mock_model]
        mock_client.chat.completions.create.side_effect = mock_responses
        mock_openai_class.return_value = mock_client

        executed_notebook_path = execute_notebook_with_params(
            ANNOTATION_NOTEBOOK_PATH,
            {},  # No parameters
            output_dir=temp_output_dir,
        )

        # Validate data pipeline integrity
        with open(executed_notebook_path, "r") as f:
            executed_notebook = json.load(f)

        # Look for data processing evidence
        data_processing_evidence = {
            "dataset_loaded": False,
            "flows_created": False,
            "data_generated": False,
            "evaluation_performed": False,
        }

        for cell in executed_notebook["cells"]:
            if cell.get("cell_type") == "code":
                cell_source = "".join(cell.get("source", []))

                # Check for dataset loading
                if "load_dataset" in cell_source or "fancyzhx/ag_news" in cell_source:
                    data_processing_evidence["dataset_loaded"] = True

                # Check for flow creation
                if "Flow.from_yaml" in cell_source:
                    data_processing_evidence["flows_created"] = True

                # Check for data generation
                if "flow.generate" in cell_source:
                    data_processing_evidence["data_generated"] = True

                # Check for evaluation
                if "classification_report" in cell_source:
                    data_processing_evidence["evaluation_performed"] = True

        # Verify all pipeline steps are present
        missing_steps = [
            step for step, found in data_processing_evidence.items() if not found
        ]
        assert len(missing_steps) == 0, f"Missing pipeline steps: {missing_steps}"
