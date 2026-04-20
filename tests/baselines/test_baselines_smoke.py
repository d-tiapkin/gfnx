"""
Smoke tests for all gfnx baseline scripts.

Each test verifies that the baseline runs end-to-end without errors using
num_train_steps=2. W&B logging is disabled (null writer) so these tests run
in CI without network access.

Phylo tests require the dataset files produced by ``download_ds.sh``. They
are automatically skipped when the files are absent.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent

# Overrides applied to every standard (single-seed) baseline.
_COMMON = [
    "num_train_steps=2",
    "logging.eval_each=1",
    "logging.track_each=1",
    "logging.tqdm_print_rate=1",
    "logging.use_writer=false",
    "writer.writer_type=null",
    "writer.save_locally=false",
]

# Overrides applied to every multiseed baseline.
_COMMON_MULTISEED = [
    "num_train_steps=2",
    "num_evals=1",
]

# (script_name, extra_overrides) for standard baselines
STANDARD_BASELINES = [
    ("tb_hypergrid", []),
    ("db_hypergrid", []),
    ("subtb_hypergrid", []),
    ("soft_dqn_hypergrid", []),
    ("db_bitseq", ["metrics.batch_size=128", "metrics.n_rounds=1"]),
    ("tb_bitseq", ["metrics.batch_size=128", "metrics.n_rounds=1"]),
    ("subtb_bitseq", ["metrics.batch_size=128", "metrics.n_rounds=1"]),
    ("db_amp", ["metrics.num_traj=32"]),
    ("tb_amp", ["metrics.num_traj=32"]),
    ("db_tfbind", []),
    ("tb_tfbind", []),
    ("subtb_tfbind", []),
    ("db_qm9_small", []),
    ("tb_qm9_small", []),
    ("tb_ising", ["data.num_samples=50"]),
    (
        "mdb_dag",
        ["metrics.batch_size=200", "metrics.n_rounds=1"],
    ),
    (
        "mdb_dag_replay_buffer",
        [
            "replay_buffer.min_length=1",
            "replay_buffer.sample_batch_size=2",
            "metrics.batch_size=200",
            "metrics.n_rounds=1",
        ],
    ),
]

MULTISEED_BASELINES = [
    ("db_hypergrid_multiseed", []),
    ("tb_hypergrid_multiseed", []),
]


def _run_baseline(
    script: str, extra: list[str], cwd: Path, timeout: int = 300
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python", f"baselines/{script}.py", *extra],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.mark.parametrize(
    "script,extra", STANDARD_BASELINES, ids=[s for s, _ in STANDARD_BASELINES]
)
def test_baseline_smoke(script, extra, tmp_path):
    log_dir = tmp_path / script
    log_dir.mkdir()
    result = _run_baseline(
        script,
        [
            *_COMMON,
            f"logging.log_dir={log_dir}",
            f"hydra.run.dir={log_dir}/hydra",
            *extra,
        ],
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"{script} failed (exit {result.returncode})\n"
        f"--- STDERR ---\n{result.stderr[-3000:]}"
    )


@pytest.mark.parametrize(
    "script,extra", MULTISEED_BASELINES, ids=[s for s, _ in MULTISEED_BASELINES]
)
def test_multiseed_baseline_smoke(script, extra, tmp_path):
    log_dir = tmp_path / script
    log_dir.mkdir()
    result = _run_baseline(
        script,
        [
            *_COMMON_MULTISEED,
            f"hydra.run.dir={log_dir}/hydra",
            *extra,
        ],
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"{script} failed (exit {result.returncode})\n"
        f"--- STDERR ---\n{result.stderr[-3000:]}"
    )


@pytest.fixture(scope="session")
def phylo_data_dir():
    """Return path to phylo datasets dir, skipping if files are absent."""
    data_dir = REPO_ROOT / "datasets"
    if not (data_dir / "DS1.json").exists():
        pytest.skip(
            "Phylogenetic datasets not found. Run `bash download_ds.sh` from the repo root" \
            " to download them."
        )
    return data_dir


def test_fldb_phylo_smoke(phylo_data_dir, tmp_path):
    log_dir = tmp_path / "fldb_phylo"
    log_dir.mkdir()
    result = _run_baseline(
        "fldb_phylo",
        [
            *_COMMON,
            f"environment.data_folder={phylo_data_dir}",
            f"logging.log_dir={log_dir}",
            f"hydra.run.dir={log_dir}/hydra",
            "agent.learning_rate.warmup_steps=0",
            "metrics.n_rounds=1",
            "metrics.n_terminal_states=2",
            "metrics.batch_size=2",
        ],
        cwd=REPO_ROOT,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"fldb_phylo failed (exit {result.returncode})\n"
        f"--- STDERR ---\n{result.stderr[-3000:]}"
    )
