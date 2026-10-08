# StoreyPath's GPU helper: the vision model alone, served on a machine with an NVIDIA
# GPU. Optional: Studio (docker/Dockerfile, the same image on every machine) works
# without it, and with it once told where it is. One Studio can use several helpers
# (the rooms of a floor spread over them), and several Studios one. No network needed
# at run time: the model is fetched before the build and baked in.
#
#   docker/fetch-vision.sh                                         # once, before building
#   docker build --platform linux/amd64 -f docker/gpu-helper/Dockerfile -t storeypath/gpu-helper .
#   docker save storeypath/gpu-helper | gzip -1 > storeypath-gpu-helper.tar.gz
#
# On the GPU machine (Linux, Docker and the NVIDIA Container Toolkit, NVIDIA driver 525
# or newer):
#
#   docker load -i storeypath-gpu-helper.tar.gz
#   docker run -d --name storeypath-gpu --gpus all -p 8105:8105 \
#       -e STOREYPATH_HELPER_KEY=<a long random key> storeypath/gpu-helper
#
# and Studio, wherever it runs: -e STOREYPATH_VISION_URL=http://<gpu machine>:8105/v1
# -e STOREYPATH_VISION_KEY=<the same key> (several helpers: their addresses separated
# by commas). docker/compose.yml runs both on one machine.
#
# It serves the OpenAI-compatible API (llama.cpp's llama-server) on port 8105, every
# call but /health needing the key: it does not start without STOREYPATH_HELPER_KEY
# (or with STOREYPATH_HELPER_OPEN=1, for a network nothing else can reach). It speaks
# plain HTTP unless given a certificate: mount it and set STOREYPATH_HELPER_CERT and
# STOREYPATH_HELPER_CERT_KEY to the files inside (Studio checks it: STOREYPATH_VISION_CA
# for one your organization made, STOREYPATH_VISION_INSECURE=1 for a self-signed one on
# a trusted network). Else keep it on a trusted network, or behind a proxy that speaks
# HTTPS. Its settings (slots, context, the model's name): docker/gpu-helper/start.sh.
# Choose the card with --gpus '"device=1"'; one with 32 GB free is enough.
#
# The engine is llama.cpp. vLLM may replace it, as measurements on the GPU machines
# decide: start.sh has its place (STOREYPATH_HELPER_ENGINE=vllm), and its image would
# be built FROM vllm/vllm-openai, with the model in vLLM's own format (it does not read
# this GGUF with its image encoder), the same start.sh, port, key and health check.
# Studio files answers by the model's name, so a different model needs its own
# STOREYPATH_HELPER_NAME.
#
# Inside: llama.cpp's llama-server built for CUDA (MIT), with OpenSSL (Apache-2.0) for
# HTTPS and NVIDIA's CUDA runtime and cuBLAS (CUDA Toolkit EULA, redistributable);
# Gemma 4 31B, 4-bit (Apache-2.0). Licences: /usr/share/storeypath/licenses.
# SymPoint-V2 is never in this image.

ARG CUDA=12.8.2
ARG UBUNTU=24.04

# ---- llama.cpp for CUDA ------------------------------------------------------------------
FROM nvidia/cuda:${CUDA}-devel-ubuntu${UBUNTU} AS llama
ARG LLAMA_CPP_VERSION=b11429
# Machine code for each GPU generation, so nothing is compiled when the image starts
# (and any CUDA 12 driver runs it): A100/A30 (80), RTX 30xx/A10/A40 (86), L4/L40/RTX
# 40xx (89), H100/H200 (90), Blackwell (120).
ARG CUDA_ARCHITECTURES="80-real;86-real;89-real;90-real;120-real"
ARG JOBS=""
# libssl-dev: llama-server can then serve HTTPS (--ssl-cert-file, --ssl-key-file)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates libssl-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "${LLAMA_CPP_VERSION}" https://github.com/ggml-org/llama.cpp /src
RUN cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" -DGGML_CUDA_NCCL=OFF \
        -DGGML_CUDA_CCCL_VERSION=v3.4.3 \
        -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
        -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DLLAMA_OPENSSL=ON \
        -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON \
    && cmake --build /build --target llama-server -j"${JOBS:-$(nproc)}"
# The server, its libraries and the CUDA runtime it links against, side by side.
RUN mkdir -p /opt/llama \
    && cp /build/bin/llama-server /opt/llama/ \
    && cp -P /build/bin/*.so* /opt/llama/ \
    && cp -P /usr/local/cuda/lib64/libcudart.so.12* /usr/local/cuda/lib64/libcublas.so.12* \
             /usr/local/cuda/lib64/libcublasLt.so.12* /opt/llama/ \
    && cp /src/LICENSE /opt/llama/LICENSE \
    && printf '%s\n' 'NVIDIA CUDA runtime and cuBLAS (libcudart, libcublas, libcublasLt), redistributed' \
         'under the CUDA Toolkit End User License Agreement: https://docs.nvidia.com/cuda/eula/' \
         > /opt/llama/CUDA-NOTICE.txt

# ---- the image -------------------------------------------------------------------------
FROM ubuntu:${UBUNTU}
# curl for the health check alone
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libssl3t64 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home --home-dir /home/helper helper
COPY --from=llama /opt/llama /opt/llama
COPY docker/models/vision/gemma-4-31B-it-Q4_K_M.gguf docker/models/vision/mmproj-F16.gguf /opt/storeypath/vision/
COPY docker/gpu-helper/start.sh /usr/local/bin/storeypath-helper
RUN mkdir -p /usr/share/storeypath/licenses \
    && cp /opt/llama/LICENSE /usr/share/storeypath/licenses/LICENSE-llama.cpp.txt \
    && cp /opt/llama/CUDA-NOTICE.txt /usr/share/storeypath/licenses/ \
    && cp /usr/share/doc/libssl3t64/copyright /usr/share/storeypath/licenses/COPYRIGHT-openssl.txt \
    && printf 'Gemma 4 model weights: Apache License 2.0, Google (https://ai.google.dev/gemma/docs/gemma_4_license).\nGGUF conversion: https://huggingface.co/unsloth/gemma-4-31B-it-GGUF\n' \
          > /usr/share/storeypath/licenses/LICENSE-vision-model.txt

ENV PATH="/opt/llama:${PATH}" \
    LD_LIBRARY_PATH="/opt/llama" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    STOREYPATH_HELPER_MODEL=/opt/storeypath/vision/gemma-4-31B-it-Q4_K_M.gguf \
    STOREYPATH_HELPER_MMPROJ=/opt/storeypath/vision/mmproj-F16.gguf

USER helper
EXPOSE 8105
# /health needs no key; 503 while the model loads (a minute or two), then 200
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m \
    CMD if [ -n "${STOREYPATH_HELPER_CERT:-}" ]; then curl -fsSk -o /dev/null https://127.0.0.1:8105/health; \
        else curl -fsS -o /dev/null http://127.0.0.1:8105/health; fi
LABEL org.opencontainers.image.title="StoreyPath GPU helper" \
      org.opencontainers.image.description="The vision model StoreyPath Studio asks about the plans, served on an NVIDIA GPU (OpenAI-compatible, with a key)" \
      org.opencontainers.image.source="https://github.com/StoreyPath/storeypath" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"
ENTRYPOINT ["storeypath-helper"]
