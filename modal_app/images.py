"""Modal image definitions for HydroLLM.

- `train_image`: verl GRPO + vLLM rollout + EF5 binary (in-process tool)
- `sft_image`:   TRL/transformers stack for SFT (legacy)
- `eval_image`:  vLLM-only baseline/eval image
"""

import modal


# ---------------------------------------------------------------------------
# verl training image: EF5 + verl + vLLM, CUDA 12.4 / Python 3.11
# ---------------------------------------------------------------------------
#
# verl's published Docker base bundles flash-attn + CUDA toolchains; we layer
# EF5 on top so simulation tools run in-process inside the trainer container.

train_image = (
    modal.Image.from_registry(
        "verlai/verl:app-verl0.5-vllm0.10.0-mcore0.13.0-te2.2",
    )
    .apt_install(
        "git", "gcc", "g++", "build-essential", "make",
        "libgeotiff-dev", "dh-autoreconf", "autotools-dev",
        "autoconf", "automake", "libtool", "pkg-config",
        "libgdal-dev", "wget",
    )
    # EF5 binary (in-process tool target)
    .run_commands(
        "git clone https://github.com/HyDROSLab/EF5.git /EF5",
        "cd /EF5 && git checkout d71621b948c9e13c4d74c4cd69e11bdf6c09c50a"
        " && autoreconf --force --install"
        " && ./configure CXXFLAGS='-Wall -O2' CFLAGS='-Wall -O2'"
        " && sed -i 's/-Werror//g' Makefile"
        " && make -j$(nproc)",
    )
    # Calibration data
    .run_commands(
        "wget -qO /tmp/data.tar.gz "
        '"https://huggingface.co/datasets/chrimerss/hydro_cali_agent_example'
        '/resolve/main/data.tar.gz"',
        "mkdir -p /app/data && tar -xzf /tmp/data.tar.gz -C /app/data",
        "rm /tmp/data.tar.gz",
        "mkdir -p /app/results /app/data/docs",
    )
    .env({
        "EF5_EXECUTABLE": "/EF5/bin/ef5",
        "PATH": "/EF5/bin:$PATH",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONPATH": "/workspace/src:/workspace",
        "RAY_DEDUP_LOGS": "0",
    })
    # Sci stack pinned for EF5 output parsing
    .pip_install(
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scipy==1.13.1",
        "pyarrow",
        "pyyaml",
        "jmespath",
        "wandb",
        "huggingface_hub",
        "hf_transfer",
    )
    # flash-attn is required by verl's FSDP workers but isn't in this base.
    # Prebuilt wheels are torch-version-specific and we don't know the exact
    # torch ABI of the base for sure, so build from source — slow (~10 min
    # on first build) but guaranteed to link correctly. Modal caches the
    # layer so this is a one-time cost.
    .pip_install(
        "packaging", "ninja", "wheel", "setuptools",
    )
    .run_commands(
        "pip uninstall -y flash-attn || true",
        "pip install flash-attn==2.7.4.post1 --no-build-isolation -v",
    )
    # Install verl from source — the verlai/verl base image ships system
    # deps (vLLM, mcore, te) but not verl itself as an importable package.
    # Match the v0.5.x branch to align with the base image's vllm/mcore pins.
    .run_commands(
        "git clone --depth 1 --branch v0.5.0 https://github.com/volcengine/verl.git /opt/verl"
        " || git clone --depth 1 https://github.com/volcengine/verl.git /opt/verl",
        "pip install -e /opt/verl",
    )
    # Mount project at /workspace (matches PYTHONPATH above)
    .add_local_dir("src", remote_path="/workspace/src")
    .add_local_dir("configs", remote_path="/workspace/configs")
    .add_local_dir("scripts", remote_path="/workspace/scripts")
    .add_local_dir("modal_app", remote_path="/workspace/modal_app")
    .add_local_file("control.txt", "/app/data/docs/control.txt")
)


# ---------------------------------------------------------------------------
# SFT image: TRL + transformers + EF5 (legacy)
# ---------------------------------------------------------------------------

sft_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git", "gcc", "g++", "build-essential", "make",
        "libgeotiff-dev", "dh-autoreconf", "autotools-dev",
        "autoconf", "automake", "libtool", "pkg-config",
        "libgdal-dev", "wget",
    )
    .run_commands(
        "git clone https://github.com/HyDROSLab/EF5.git /EF5",
        "cd /EF5 && git checkout d71621b948c9e13c4d74c4cd69e11bdf6c09c50a"
        " && autoreconf --force --install"
        " && ./configure CXXFLAGS='-Wall -O2' CFLAGS='-Wall -O2'"
        " && sed -i 's/-Werror//g' Makefile"
        " && make -j$(nproc)",
    )
    .run_commands(
        "wget -qO /tmp/data.tar.gz "
        '"https://huggingface.co/datasets/chrimerss/hydro_cali_agent_example'
        '/resolve/main/data.tar.gz"',
        "mkdir -p /app/data && tar -xzf /tmp/data.tar.gz -C /app/data",
        "rm /tmp/data.tar.gz",
        "mkdir -p /app/results /app/data/docs",
    )
    .env({
        "EF5_EXECUTABLE": "/EF5/bin/ef5",
        "PATH": "/EF5/bin:$PATH",
        "VLLM_USE_V1": "0",
        "TORCH_COMPILE_DISABLE": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
    .pip_install("torch>=2.4")
    .pip_install(
        "trl>=1.0",
        "transformers>=4.45",
        "accelerate",
        "peft",
        "datasets",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scipy==1.13.1",
        "wandb",
        "pyyaml",
        "jmespath",
        "huggingface_hub",
        "hf_transfer",
    )
    .add_local_dir("src/hydrollm", remote_path="/root/hydrollm")
    .add_local_dir("modal_app", remote_path="/root/modal_app")
    .add_local_file("control.txt", "/app/data/docs/control.txt")
    .add_local_dir("configs", "/app/configs")
    .add_local_dir("scripts", "/app/scripts")
    .add_local_dir("data", "/app/data/sft")
)


# ---------------------------------------------------------------------------
# Lightweight evaluation image (for baseline, no heavy training deps)
# ---------------------------------------------------------------------------

eval_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git", "gcc", "g++", "build-essential", "make",
        "libgeotiff-dev", "dh-autoreconf", "autotools-dev",
        "autoconf", "automake", "libtool", "pkg-config",
        "libgdal-dev", "wget",
    )
    .run_commands(
        "git clone https://github.com/HyDROSLab/EF5.git /EF5",
        "cd /EF5 && git checkout d71621b948c9e13c4d74c4cd69e11bdf6c09c50a"
        " && autoreconf --force --install"
        " && ./configure CXXFLAGS='-Wall -O2' CFLAGS='-Wall -O2'"
        " && sed -i 's/-Werror//g' Makefile"
        " && make -j$(nproc)",
    )
    .run_commands(
        "wget -qO /tmp/data.tar.gz "
        '"https://huggingface.co/datasets/chrimerss/hydro_cali_agent_example'
        '/resolve/main/data.tar.gz"',
        "mkdir -p /app/data && tar -xzf /tmp/data.tar.gz -C /app/data",
        "rm /tmp/data.tar.gz",
        "mkdir -p /app/results /app/data/docs",
    )
    .env({
        "EF5_EXECUTABLE": "/EF5/bin/ef5",
        "PATH": "/EF5/bin:$PATH",
        "VLLM_USE_V1": "0",  # V1 engine has torch.compile SymInt bug with Qwen2
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
    .uv_pip_install(
        "vllm>=0.6",
        "transformers>=4.45",
        "torch>=2.4",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scipy==1.13.1",
        "pyyaml",
        "hf_transfer",
    )
    .add_local_dir("src/hydrollm", remote_path="/root/hydrollm")
    .add_local_dir("modal_app", remote_path="/root/modal_app")
    .add_local_file("control.txt", "/app/data/docs/control.txt")
    .add_local_dir("configs", "/app/configs")
)
