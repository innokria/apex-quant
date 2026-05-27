
import os
import subprocess
import threading
from datetime import datetime

import gradio as gr
from huggingface_hub import snapshot_download, HfApi

# -----------------------------
# CONFIG
# -----------------------------
APEX_DIR = os.environ.get("APEX_DIR", "/app/apex-quant")
LLAMA_DIR = os.environ.get("LLAMA_DIR", "/app/llama.cpp")
HF_TOKEN = os.getenv("HF_TOKEN")

# faster + cleaner HF downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

api = HfApi()

# -----------------------------
# LOG BUFFER
# -----------------------------
LOG_BUFFER = []
JOB_STATUS = {"running": False, "log": ""}


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")

    line = f"[{ts}] {msg}"

    print(line, flush=True)

    LOG_BUFFER.append(line)

    if len(LOG_BUFFER) > 5000:
        del LOG_BUFFER[:1000]

    JOB_STATUS["log"] = "\n".join(LOG_BUFFER)

    return JOB_STATUS["log"]


# -----------------------------
# STREAM RUN
# -----------------------------
def run_stream(cmd, cwd=None, env=None):

    log("▶ RUN: " + " ".join(cmd))

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    output = ""

    for line in process.stdout:

        if line:

            line = line.rstrip()

            print(line, flush=True)

            LOG_BUFFER.append(line)

            JOB_STATUS["log"] = "\n".join(LOG_BUFFER)

            output += line + "\n"

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(output)

    return output


# -----------------------------
# DOWNLOAD MODEL (LIVE LOGS)
# -----------------------------
def download_model(repo_id):

    log(f"📥 Downloading: {repo_id}")

    # cleanup old model first
    if os.path.exists("/tmp/model"):

        log("🧹 removing old /tmp/model")

        run_stream([
            "rm",
            "-rf",
            "/tmp/model"
        ])

    os.makedirs("/tmp/model", exist_ok=True)

    log("📡 Starting snapshot_download...")
    log("⏳ Large models can take a LONG time")
    log("⏳ 27B/32B models may need 30-90 mins")

    path = snapshot_download(
        repo_id=repo_id,
        local_dir="/tmp/model",
        resume_download=True,
        max_workers=2,
        tqdm_class=None
    )

    log("✅ Download complete")

    # show downloaded files
    total_size = 0

    for root, dirs, files in os.walk(path):

        for f in files:

            fp = os.path.join(root, f)

            try:

                sz = os.path.getsize(fp)

                total_size += sz

                log(
                    f"📄 {f} | "
                    f"{sz / (1024**3):.2f} GB"
                )

            except:
                pass

    log(
        f"📦 TOTAL SIZE: "
        f"{total_size / (1024**3):.2f} GB"
    )

    return path


# -----------------------------
# LLAMA SETUP
# -----------------------------
def ensure_llama():

    if os.path.exists(LLAMA_DIR):
        return

    log("📥 cloning llama.cpp")

    run_stream([
        "git",
        "clone",
        "--depth",
        "1",
        "https://github.com/ggerganov/llama.cpp",
        LLAMA_DIR
    ])


# -----------------------------
# BUILD LLAMA
# -----------------------------
def build_llama():

    ensure_llama()

    build_dir = os.path.join(
        LLAMA_DIR,
        "build"
    )

    bin_dir = os.path.join(
        build_dir,
        "bin"
    )

    if os.path.exists(
        os.path.join(
            bin_dir,
            "llama-quantize"
        )
    ):
        log("♻️ llama.cpp already built")
        return bin_dir

    log("🔧 building llama.cpp")

    run_stream(
        [
            "cmake",
            "-B",
            "build",
            "-DLLAMA_BUILD_TOOLS=ON"
        ],
        cwd=LLAMA_DIR
    )

    run_stream(
        [
            "cmake",
            "--build",
            "build",
            "-j",
            "--target",
            "llama-quantize"
        ],
        cwd=LLAMA_DIR
    )

    return bin_dir


# -----------------------------
# CONVERTER
# -----------------------------
def find_converter():

    ensure_llama()

    candidates = [

        os.path.join(
            LLAMA_DIR,
            "convert_hf_to_gguf.py"
        ),

        os.path.join(
            LLAMA_DIR,
            "scripts",
            "convert_hf_to_gguf.py"
        ),

        os.path.join(
            LLAMA_DIR,
            "tools",
            "convert_hf_to_gguf.py"
        ),

        os.path.join(
            LLAMA_DIR,
            "convert.py"
        ),
    ]

    for c in candidates:

        if os.path.exists(c):
            return c

    raise RuntimeError("No converter found")


# -----------------------------
# HF → GGUF
# -----------------------------
def build_f16(model_dir):

    log("🧠 HF → F16 START")

    script = find_converter()

    out = os.path.join(
        APEX_DIR,
        "model-f16.gguf"
    )

    # remove old file
    if os.path.exists(out):

        log("🧹 removing old F16")

        os.remove(out)

    run_stream([
        "python3",
        script,
        model_dir,
        "--outtype",
        "f16",
        "--outfile",
        out
    ])

    if not os.path.exists(out):
        raise RuntimeError("❌ F16 generation failed")

    size = os.path.getsize(out) / (1024**3)

    log(f"✅ F16 CREATED: {size:.2f} GB")

    return out


# -----------------------------
# QUANTIZE
# -----------------------------
def quantize(f16_path, profile):

    log(f"⚙️ QUANT: {profile}")

    # remove broken system binaries
    for p in [
        "/usr/local/bin/llama-quantize",
        "/usr/bin/llama-quantize"
    ]:

        if os.path.exists(p):

            try:
                os.remove(p)
                log(f"🧹 removed broken binary: {p}")
            except:
                pass

    build_llama()

    env = os.environ.copy()

    env["PATH"] = (
        os.path.join(LLAMA_DIR, "build", "bin")
        + ":"
        + env.get("PATH", "")
    )

    out = os.path.join(
        APEX_DIR,
        f"model-{profile}.gguf"
    )

    # remove old output
    if os.path.exists(out):

        log("🧹 removing old quant")

        os.remove(out)

    quant_script = os.path.join(
        APEX_DIR,
        "scripts",
        "quantize.sh"
    )

    log("🚀 starting APEX quantization")

    run_stream([
        "bash",
        quant_script,
        "--profile",
        profile,
        f16_path,
        out
    ], cwd=APEX_DIR, env=env)

    if not os.path.exists(out):
        raise RuntimeError("❌ quantization failed")

    size = os.path.getsize(out) / (1024**3)

    log(f"✅ QUANT COMPLETE: {size:.2f} GB")

    return out


# -----------------------------
# WORKER
# -----------------------------
def worker(source_repo, profile, target_repo):

    try:

        JOB_STATUS["running"] = True

        LOG_BUFFER.clear()

        log("🚀 START PIPELINE")

        model_dir = download_model(source_repo)

        f16 = build_f16(model_dir)

        gguf = quantize(f16, profile)

        size = os.path.getsize(gguf) / (1024**3)

        log(f"📦 FINAL SIZE: {size:.2f} GB")

        log("📤 uploading...")

        api.create_repo(
            target_repo,
            repo_type="model",
            exist_ok=True,
            token=HF_TOKEN
        )

        upload_name = os.path.basename(gguf)

        api.upload_file(
            path_or_fileobj=gguf,
            path_in_repo=upload_name,
            repo_id=target_repo,
            repo_type="model",
            token=HF_TOKEN
        )

        log("🎉 DONE")

    except Exception as e:

        log(f"❌ ERROR: {str(e)}")

    finally:

        JOB_STATUS["running"] = False


# -----------------------------
# LAUNCH
# -----------------------------
def pipeline(source_repo, profile, target_repo):

    if JOB_STATUS["running"]:
        return "⚠️ Already running"

    thread = threading.Thread(
        target=worker,
        args=(source_repo, profile, target_repo),
        daemon=True
    )

    thread.start()

    return "🚀 Job started"


# -----------------------------
# LOG VIEWER
# -----------------------------
def get_logs():
    return JOB_STATUS.get("log", "")


# -----------------------------
# UI
# -----------------------------
with gr.Blocks() as demo:

    gr.Markdown(
        "# ⚡ GGUF Factory "
        "(Large Model Ready)"
    )

    source = gr.Textbox(
        value="rahul7star/gemma-4-finetune",
        label="HF Source Repo"
    )

    profile = gr.Dropdown(
        [
            "i-quality",
            "quality",
            "balanced",
            "compact"
        ],
        value="i-quality",
        label="Profile"
    )

    target = gr.Textbox(
        value="rahul7star/gemma-gguf",
        label="HF Output Repo"
    )

    btn = gr.Button("🚀 Start")

    out = gr.Textbox(
        label="Status"
    )

    logs = gr.Textbox(
        label="Logs",
        lines=35,
        autoscroll=True
    )

    btn.click(
        pipeline,
        [source, profile, target],
        out
    )

    demo.load(
        get_logs,
        None,
        logs
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True
)
