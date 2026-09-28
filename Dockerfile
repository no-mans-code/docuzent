# Multi-stage build: Rust binaries in one stage, Python/Docling runtime in
# the other. Exists specifically so Docling's own dependency chain
# (torch/transformers) runs inside a container's own Linux environment,
# entirely separate from a Windows host's Smart App Control policy, which
# has been observed to persistently block docling.exe (and, one layer
# deeper, torch.dll itself) on at least one real dev machine - see
# https://github.com/no-mans-code/docuzent/issues/23 for the full writeup.

# ---- Stage 1: build the Rust binaries ----
FROM rust:1-slim-bookworm AS builder
WORKDIR /app

# redb (via kvcache) and a couple of transitive deps need a C toolchain;
# git is needed for the two git dependencies (kvcache, ollama-kv-profiler).
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY docuzent-core ./docuzent-core
COPY docuzent-cli ./docuzent-cli
COPY docuzent-web ./docuzent-web

RUN cargo build --release -p docuzent-web -p docuzent-cli

# ---- Stage 2: runtime - Python + Docling, plus the built Rust binaries ----
FROM python:3.11-slim-bookworm AS runtime
WORKDIR /app

# opencv-python (pulled in transitively by docling-ibm-models' table
# structure model) needs real shared libraries a minimal Debian slim
# image doesn't ship - confirmed via a real failure: `import cv2` raised
# `ImportError: libxcb.so.1: cannot open shared object file`. This is the
# standard fix set for opencv-python on slim Debian, not something
# specific to this project.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libxcb1 libxext6 libsm6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# `docling` lands on PATH via the pip install above - ingest.rs's
# resolve_docling_bin() already falls back to bare "docling" on PATH when
# no .venv is found, so no DOCLING_BIN override or venv setup is needed
# inside the (already isolated) container.
COPY --from=builder /app/target/release/docuzent-web /usr/local/bin/docuzent-web
COPY --from=builder /app/target/release/docuzent /usr/local/bin/docuzent

# On-disk caches (LLM-context + Docling parse) - mount this as a volume
# to survive container restarts instead of starting cold every run.
VOLUME ["/data"]

EXPOSE 3800

# host.docker.internal is Docker Desktop's own DNS name for the host
# machine (Windows/Mac) - reaches a real `ollama serve` running on the
# host, since "localhost" from inside the container means the container
# itself, not the host. Model and cache mode are runtime-switchable from
# the UI regardless of what's passed here as the initial pick.
ENTRYPOINT ["docuzent-web"]
CMD ["--host", "http://host.docker.internal:11434", \
     "--port", "3800", \
     "--model", "qwen2.5:3b", \
     "--mode", "adaptive", \
     "--cache", "/data/context.redb", \
     "--docling-cache", "/data/docling.redb"]
