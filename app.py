import os
import subprocess
from datetime import datetime

import gradio as gr
from huggingface_hub import snapshot_download, HfApi

# -----------------------------
# PATHS
# -----------------------------
APEX_DIR = os.environ.get("APEX_DIR", "/app/apex-quant")
LLAMA_DIR = os.environ.get("LLAMA_DIR", "/app/llama.cpp")

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi()

# -----------------------------
# LOGGING
# -----------------------------
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    return line + "\n"

# -----------------------------
# RUN SHELL
# -----------------------------
def run(cmd, cwd=None, env=None):
    log("▶ RUN: " + " ".join(cmd))

    p = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        env=env
    )

    if p.stdout:
        log(p.stdout)

    if p.stderr:
        log("STDERR:\n" + p.stderr)

    if p.returncode != 0:
        raise RuntimeError(p.stderr)

    return p.stdout

# -----------------------------
# ENSURE LLAMA.CPP EXISTS
# -----------------------------
def ensure_llama_cpp():
    if os.path.exists(LLAMA_DIR):
        return LLAMA_DIR

    log("📥 Cloning llama.cpp...")
    run([
        "git",
        "clone",
        "--depth", "1",
        "https://github.com/ggerganov/llama.cpp",
        LLAMA_DIR
    ])

    return LLAMA_DIR

# -----------------------------
# BUILD LLAMA.CPP (🔥 FIX)
# -----------------------------
def build_llama_cpp():
    log("🔧 Building llama.cpp (required for quantization)")

    ensure_llama_cpp()

    # clean build dir
    build_dir = os.path.join(LLAMA_DIR, "build")

    run(["cmake", "-B", "build"], cwd=LLAMA_DIR)
    run(["cmake", "--build", "build", "-j"], cwd=LLAMA_DIR)

    bin_path = os.path.join(build_dir, "bin")
    log(f"✅ llama.cpp built: {bin_path}")

    return bin_path

# -----------------------------
# DOWNLOAD MODEL
# -----------------------------
def download_model(repo_id):
    log(f"📥 Downloading HF model: {repo_id}")

    path = snapshot_download(
        repo_id=repo_id,
        local_dir="/tmp/model",
        local_dir_use_symlinks=False
    )

    log(f"✅ Download complete: {path}")
    return path

# -----------------------------
# FIND CONVERTER
# -----------------------------
def find_converter():
    ensure_llama_cpp()

    candidates = [
        os.path.join(LLAMA_DIR, "convert_hf_to_gguf.py"),
        os.path.join(LLAMA_DIR, "convert-hf-to-gguf.py"),
        os.path.join(LLAMA_DIR, "scripts", "convert_hf_to_gguf.py"),
        os.path.join(LLAMA_DIR, "convert.py"),
    ]

    for c in candidates:
        if os.path.exists(c):
            log(f"🔧 Found converter: {c}")
            return c

    raise RuntimeError("❌ No HF→GGUF converter found in llama.cpp")

# -----------------------------
# HF → GGUF
# -----------------------------
def build_f16(model_dir):
    log("🧠 STEP: HF → F16 GGUF")

    script = find_converter()
    f16_path = os.path.join(APEX_DIR, "model-f16.gguf")

    run([
        "python3",
        script,
        model_dir,
        "--outtype", "f16",
        "--outfile", f16_path
    ])

    if not os.path.exists(f16_path):
        raise RuntimeError("❌ F16 GGUF generation failed")

    log(f"✅ F16 CREATED: {f16_path}")
    return f16_path

# -----------------------------
# VALID PROFILES
# -----------------------------
VALID_PROFILES = {
    "quality",
    "i-quality",
    "balanced",
    "i-balanced",
    "compact",
    "i-compact",
    "mini",
    "full-pipeline"
}

# -----------------------------
# FIX BROKEN BINARIES
# -----------------------------
def fix_bad_binaries():
    bad = [
        "/usr/local/bin/llama-quantize",
        "/usr/bin/llama-quantize"
    ]

    for p in bad:
        if os.path.exists(p):
            try:
                os.remove(p)
                log(f"🧹 Removed broken binary: {p}")
            except:
                log(f"⚠️ Could not remove: {p}")

# -----------------------------
# QUANTIZE (🔥 FIXED)
# -----------------------------
def quantize(f16_path, profile):
    log(f"⚙️ QUANTIZE: {profile}")

    if profile not in VALID_PROFILES:
        raise RuntimeError("❌ Invalid profile")

    # 🔥 IMPORTANT FIX
    fix_bad_binaries()
    bin_path = build_llama_cpp()

    script = os.path.join(APEX_DIR, "scripts/quantize.sh")

    if not os.path.exists(script):
        raise RuntimeError("❌ quantize.sh missing in apex-quant")

    out_path = os.path.join(APEX_DIR, f"model-apex-{profile}.gguf")

    env = os.environ.copy()
    env["PATH"] = bin_path + ":" + env.get("PATH", "")

    run([
        "bash",
        script,
        "--profile",
        profile,
        f16_path,
        out_path
    ], cwd=APEX_DIR, env=env)

    if not os.path.exists(out_path):
        raise RuntimeError("❌ Quantization failed")

    log(f"✅ OUTPUT: {out_path}")
    return out_path

# -----------------------------
# FULL PIPELINE
# -----------------------------
def full_pipeline(source_repo):
    log("🚀 START FULL PIPELINE")

    model_dir = download_model(source_repo)
    f16 = build_f16(model_dir)
    gguf = quantize(f16, "i-quality")

    return gguf

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def pipeline(source_repo, profile, target_repo):

    try:
        log("========================================")
        log("🚀 GGUF FACTORY START")
        log(f"📦 SOURCE: {source_repo}")
        log(f"🎯 PROFILE: {profile}")
        log(f"📤 TARGET: {target_repo}")
        log("========================================")

        if profile == "full-pipeline":
            gguf = full_pipeline(source_repo)
        else:
            model = download_model(source_repo)
            f16 = build_f16(model)
            gguf = quantize(f16, profile)

        if not HF_TOKEN:
            return "❌ HF_TOKEN missing"

        log(f"📤 Uploading → {target_repo}")

        api.create_repo(
            target_repo,
            repo_type="model",
            exist_ok=True,
            token=HF_TOKEN
        )

        api.upload_file(
            path_or_fileobj=gguf,
            path_in_repo=os.path.basename(gguf),
            repo_id=target_repo,
            repo_type="model",
            token=HF_TOKEN
        )

        log("✅ Upload complete")
        return f"✅ DONE → {target_repo}"

    except Exception as e:
        log(f"❌ ERROR: {str(e)}")
        return f"❌ ERROR: {str(e)}"

# -----------------------------
# UI
# -----------------------------
with gr.Blocks() as demo:

    gr.Markdown("# ⚡ GGUF Factory (FIXED QUANT BUILD)")

    source = gr.Textbox(
        label="HF Source Repo",
        value="rahul7star/gemma-4-finetune"
    )

    profile = gr.Dropdown(
        [
            "quality",
            "i-quality",
            "balanced",
            "i-balanced",
            "compact",
            "i-compact",
            "mini",
            "full-pipeline"
        ],
        value="i-quality",
        label="Profile"
    )

    target = gr.Textbox(
        label="HF Output Repo",
        value="rahul7star/gemma-gguf"
    )

    btn = gr.Button("🚀 Run")
    out = gr.Textbox(label="Logs", lines=30)

    btn.click(
        pipeline,
        [source, profile, target],
        out
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
