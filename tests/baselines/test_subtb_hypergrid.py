"""
Integration tests for subtb_hypergrid baseline script.
"""

import ast
import re
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple

import pytest


class TrainingResult(NamedTuple):
    """Container for training run results."""
    returncode: int
    stdout: str
    stderr: str


@pytest.fixture(scope="module")
def training_paths():
    """Provide paths needed for running the baseline script."""
    root = Path(__file__).parent.parent.parent
    script_path = root / "baselines" / "subtb_hypergrid.py"
    config_dir = root / "tests" / "baselines" / "configs"
    baselines_dir = root / "baselines"
    output_base = root / "tmp"
    
    return {
        "script": script_path,
        "config_dir": config_dir,
        "baselines_dir": baselines_dir,
        "output_base": output_base,
    }


@pytest.fixture(scope="module")
def training_result(training_paths) -> TrainingResult:
    """
    Run the training script once and return results.
    """
    result = subprocess.run(
        [
            "python",
            str(training_paths["script"]),
            "--config-path", str(training_paths["config_dir"].absolute()),
            "--config-name", "test_subtb_hypergrid",
            f"hydra.run.dir={training_paths['output_base'].absolute()!s}",
            f"hydra.sweep.dir={training_paths['output_base'].absolute()!s}",
        ],
        cwd=training_paths["baselines_dir"],  # Required for relative imports in script
        capture_output=True,
        text=True,
        timeout=600,  # 10 minute timeout
        check=False,
    )
    
    return TrainingResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


@pytest.fixture(scope="module", autouse=True)
def cleanup_training_output(training_paths, training_result):
    """Cleanup training output directories after all tests complete."""
    yield  # Tests run here
    
    # Cleanup after all tests
    output_base = training_paths["output_base"]
    if output_base.exists():
        print(f"\nCleaning up test output directory: {output_base}")
        shutil.rmtree(output_base, ignore_errors=True)


def test_script_runs_successfully(training_result):
    """Verify the training script completes without errors."""
    assert training_result.returncode == 0, (
        f"Script failed with return code {training_result.returncode}\n"
        f"STDERR:\n{training_result.stderr}"
    )


def parse_last_training_loss(stdout: str) -> float:
    """Parse the last training loss from the stdout."""
    dict_pattern = r"\[.*?\]\[.*?\]\[.*?\] - (\{.*?\})"
    matches = list(re.finditer(dict_pattern, stdout, re.MULTILINE))
    if not matches:
        raise ValueError("No training metrics found in output")
    for match in reversed(matches):
        dict_str = match.group(1)
        try:
            metrics_dict = ast.literal_eval(dict_str)
            if 'train/mean_loss' in metrics_dict:
                return float(metrics_dict['train/mean_loss'])
        except (ValueError, SyntaxError):
            continue
    
    raise ValueError("No 'train/mean_loss' metric found in output")


def test_training_metrics(training_result, training_paths):
    """Verify that final training loss is below the configured threshold."""
    from omegaconf import OmegaConf
    
    config_path = training_paths["config_dir"] / "test_subtb_hypergrid.yaml"
    config = OmegaConf.load(config_path)
    loss_threshold = config.loss_threshold
    
    stdout = training_result.stdout
    final_loss = parse_last_training_loss(stdout)
    
    assert final_loss < loss_threshold, (
        f"Final training loss {final_loss:.6e} exceeds threshold {loss_threshold:.6e}"
    )