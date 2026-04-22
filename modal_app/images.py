"""Modal image definitions for HydroLLM.

Defines a single image that combines:
1. EF5/CREST hydrologic model (compiled from source)
2. Hydro calibration data (downloaded from HuggingFace)
3. Training stack: Unsloth + TRL + vLLM + PyTorch
"""

import modal

# ---------------------------------------------------------------------------
# Combined training + simulation image
# ---------------------------------------------------------------------------

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    # ----- System dependencies for EF5 compilation -----
    .apt_install(
        "git",
        "gcc",
        "g++",
        "build-essential",
        "make",
        "libgeotiff-dev",
        "dh-autoreconf",
        "autotools-dev",
        "autoconf",
        "automake",
        "libtool",
        "pkg-config",
        "libgdal-dev",
        "wget",
    )
    # ----- Build EF5 from source (pinned commit) -----
    .run_commands(
        "git clone https://github.com/HyDROSLab/EF5.git /EF5",
        "cd /EF5 && git checkout d71621b948c9e13c4d74c4cd69e11bdf6c09c50a"
        " && autoreconf --force --install"
        " && ./configure CXXFLAGS='-Wall -O2' CFLAGS='-Wall -O2'"
        " && sed -i 's/-Werror//g' Makefile"
        " && make -j$(nproc)",
    )
    # ----- Download hydro calibration data -----
    .run_commands(
        "wget -qO /tmp/data.tar.gz "
        '"https://huggingface.co/datasets/chrimerss/hydro_cali_agent_example'
        '/resolve/main/data.tar.gz"',
        "mkdir -p /app/data && tar -xzf /tmp/data.tar.gz -C /app/data",
        "rm /tmp/data.tar.gz",
        "mkdir -p /app/results /app/data/docs",
    )
    # ----- Environment variables -----
    .env({
        "EF5_EXECUTABLE": "/EF5/bin/ef5",
        "PATH": "/EF5/bin:$PATH",
        "VLLM_USE_V1": "0",
        "TORCH_COMPILE_DISABLE": "1",  # Prevents SymInt crash in vLLM model loading
    })
    # ----- Python training stack -----
    # Stage 1: PyTorch (pins CUDA/nvidia library versions)
    .pip_install(
        "torch>=2.4",
    )
    # Stage 2: Training stack
    .pip_install(
        "trl>=1.0",
        "transformers>=4.45",
        "accelerate",
        "peft",
        "datasets",
        # Scientific computing (pinned for EF5 compatibility)
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scipy==1.13.1",
        # Utilities
        "wandb",
        "pyyaml",
        "jmespath",
    )
    # ----- Add project source code -----
    .add_local_dir("src/hydrollm", remote_path="/root/hydrollm")
    .add_local_dir("modal_app", remote_path="/root/modal_app")
    .add_local_file("control.txt", "/app/data/docs/control.txt")
    .add_local_dir("configs", "/app/configs")
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
    })
    .uv_pip_install(
        "vllm>=0.6",
        "transformers>=4.45",
        "torch>=2.4",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scipy==1.13.1",
        "pyyaml",
    )
    .add_local_dir("src/hydrollm", remote_path="/root/hydrollm")
    .add_local_dir("modal_app", remote_path="/root/modal_app")
    .add_local_file("control.txt", "/app/data/docs/control.txt")
    .add_local_dir("configs", "/app/configs")
)
