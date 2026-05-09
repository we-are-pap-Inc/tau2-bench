"""
Modal runner for τ³ / τ-Voice StageGate experiments.

This is a template. Codex must verify paths and Modal SDK syntax against the current repo and Modal version before final use.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "tau3-voice-stagegate"
REPO_URL = os.environ.get(
    "STAGEGATE_REPO_URL", "https://github.com/YOUR_ORG/tau2-bench.git"
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("tau3-voice-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "curl",
        "ca-certificates",
        "build-essential",
        "pkg-config",
        "ffmpeg",
        "portaudio19-dev",
        "libsndfile1",
        "libsndfile1-dev",
        "jq",
    )
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env(
        {
            "PATH": "/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        }
    )
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("tau3-voice-secrets")],
    volumes={"/runs": volume},
    timeout=24 * 60 * 60,
    cpu=4,
    memory=16_384,
)
def run_domain(
    condition: str, domain: str, repo_ref: str, batch_id: str
) -> dict[str, str]:
    """Run one condition/domain pair.

    condition: baseline | stage_only | stagegate
    domain: retail | airline | telecom
    repo_ref: fixed git commit SHA or branch
    batch_id: stable ID for this experiment batch
    """

    if condition not in {"baseline", "stage_only", "stagegate"}:
        raise ValueError(f"Unknown condition: {condition}")
    if domain not in {"retail", "airline", "telecom"}:
        raise ValueError(f"Unknown domain: {domain}")

    workdir = Path("/tmp/tau2-bench")
    save_name = f"{batch_id}_{condition}_{domain}"
    artifact_root = Path(f"/runs/{batch_id}/{condition}/{domain}")
    artifact_root.mkdir(parents=True, exist_ok=True)

    subprocess.run(["git", "clone", REPO_URL, str(workdir)], check=True)
    subprocess.run(["git", "checkout", repo_ref], cwd=workdir, check=True)
    subprocess.run(["git", "rev-parse", "HEAD"], cwd=workdir, check=True)

    subprocess.run(
        ["uv", "sync", "--extra", "voice", "--extra", "dev"], cwd=workdir, check=True
    )
    subprocess.run(["uv", "run", "tau2", "check-data"], cwd=workdir, check=True)

    env = os.environ.copy()
    env["TAU2_STAGEGATE_CONDITION"] = condition
    env["TAU2_TRACE_JSONL"] = str(artifact_root / "trace_events.jsonl")

    cmd = [
        "uv",
        "run",
        "tau2",
        "run",
        "--domain",
        domain,
        "--audio-native",
        "--audio-native-provider",
        "openai",
        "--audio-native-model",
        "gpt-realtime-2",
        "--reasoning-effort",
        "high",
        "--speech-complexity",
        "regular",
        "--tick-duration",
        "0.2",
        "--max-steps-seconds",
        "1200",
        "--max-concurrency",
        "1",
        "--seed",
        "300",
        "--verbose-logs",
        "--auto-resume",
        "--save-to",
        save_name,
    ]

    subprocess.run(cmd, cwd=workdir, env=env, check=True)

    simulation_dir = workdir / "data" / "simulations" / save_name
    if simulation_dir.exists():
        subprocess.run(
            ["bash", "-lc", f"cp -R {simulation_dir} {artifact_root}/"], check=True
        )
    else:
        raise FileNotFoundError(f"Expected simulation dir not found: {simulation_dir}")

    volume.commit()

    return {
        "condition": condition,
        "domain": domain,
        "save_name": save_name,
        "artifact_dir": str(artifact_root),
    }


@app.local_entrypoint()
def launch(batch_id: str, repo_ref: str = "main", smoke: bool = False):
    """Launch the experiment matrix.

    Use smoke=true to launch only stagegate/retail for infrastructure testing.
    """

    if smoke:
        jobs = [("stagegate", "retail")]
    else:
        jobs = [
            (condition, domain)
            for condition in ["baseline", "stage_only", "stagegate"]
            for domain in ["retail", "airline", "telecom"]
        ]

    calls = []
    for condition, domain in jobs:
        call = run_domain.spawn(condition, domain, repo_ref, batch_id)
        calls.append(call)
        print(
            {
                "condition": condition,
                "domain": domain,
                "function_call_id": call.object_id,
            }
        )

    print(f"Launched {len(calls)} jobs")
