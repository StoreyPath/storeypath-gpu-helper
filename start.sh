#!/bin/sh
# StoreyPath's GPU helper: the vision model server, OpenAI-compatible, on port 8105.
# Arguments are passed on to the engine (llama-server) as they are.
#
#   STOREYPATH_HELPER_KEY       the key every call must carry (Studio sends it: its
#                               STOREYPATH_VISION_KEY). Without one the helper does not
#                               start, unless STOREYPATH_HELPER_OPEN=1 (a network
#                               nothing else can reach)
#   STOREYPATH_HELPER_SLOTS     questions answered at once (default 2); Studio's
#                               STOREYPATH_VISION_PARALLEL should match it
#   STOREYPATH_HELPER_CONTEXT   tokens of context per question (default 8192)
#   STOREYPATH_HELPER_CERT      HTTPS with this certificate (PEM, its chain after it)
#   STOREYPATH_HELPER_CERT_KEY  and its private key (PEM); without them, plain HTTP: for
#                               a trusted network, or behind a proxy that speaks HTTPS
#   STOREYPATH_HELPER_NAME      the name the model is served under (default: the model
#                               file's, gemma-4-31B-it-Q4_K_M). Studio files each answer
#                               by it: helpers serving the same model use the same name
#   STOREYPATH_HELPER_MODEL, STOREYPATH_HELPER_MMPROJ
#                               the model and its image encoder (the image sets both)
#   STOREYPATH_HELPER_ENGINE    llama.cpp (the default, in this image); vllm: where vLLM
#                               would go, if measurements choose it (see the Dockerfile)
#
# One GPU with 32 GB free runs the model with two questions at once (each one more
# needs more memory). It takes a minute or two to load: /health answers 200 once it
# has (503 while loading), and Studio leaves the helper out until then.
set -eu
port=8105
slots="${STOREYPATH_HELPER_SLOTS:-2}"
context="${STOREYPATH_HELPER_CONTEXT:-8192}"
key="${STOREYPATH_HELPER_KEY:-}"
model="${STOREYPATH_HELPER_MODEL:?STOREYPATH_HELPER_MODEL: the model to serve}"
name="${STOREYPATH_HELPER_NAME:-$(basename "$model" .gguf)}"
cert="${STOREYPATH_HELPER_CERT:-}"
cert_key="${STOREYPATH_HELPER_CERT_KEY:-}"
engine="${STOREYPATH_HELPER_ENGINE:-llama.cpp}"
unset STOREYPATH_HELPER_KEY

if [ -z "$key" ] && [ "${STOREYPATH_HELPER_OPEN:-}" != 1 ]; then
    echo "no STOREYPATH_HELPER_KEY: give the helper a key, the one Studio sends as STOREYPATH_VISION_KEY" >&2
    echo "(e.g. -e STOREYPATH_HELPER_KEY=\$(openssl rand -hex 32)), or STOREYPATH_HELPER_OPEN=1 to serve without one" >&2
    exit 2
fi
scheme=http
if [ -n "$cert$cert_key" ]; then
    if [ ! -r "$cert" ] || [ ! -r "$cert_key" ]; then
        echo "STOREYPATH_HELPER_CERT and STOREYPATH_HELPER_CERT_KEY: both must name readable PEM files" >&2
        exit 2
    fi
    scheme=https
fi
echo "serving $name ($engine) at $scheme://<this machine>:$port/v1: $slots question(s) at once, $context tokens each," \
     "$(if [ -n "$key" ]; then echo "a key needed"; else echo "OPEN: no key needed"; fi)"

case "$engine" in
llama.cpp)
    if [ -n "$cert" ]; then
        set -- --ssl-cert-file "$cert" --ssl-key-file "$cert_key" "$@"
    fi
    # the key from the environment, not the command line (which the host's ps shows)
    if [ -n "$key" ]; then
        LLAMA_API_KEY="$key"
        export LLAMA_API_KEY
    fi
    exec llama-server -m "$model" --mmproj "${STOREYPATH_HELPER_MMPROJ:?STOREYPATH_HELPER_MMPROJ: the image encoder}" \
        --alias "$name" --host 0.0.0.0 --port "$port" -ngl 99 \
        -c "$((context * slots))" -np "$slots" -b 4096 -ub 4096 \
        --image-min-tokens 1024 --jinja --no-webui --no-slots "$@"
    ;;
vllm)
    # vLLM's place, for an image built FROM vllm/vllm-openai with the model in vLLM's
    # own format (a folder of safetensors; vLLM does not read this GGUF and its
    # encoder) and STOREYPATH_HELPER_NAME set: a different model, a different name.
    if [ -n "$cert" ]; then
        set -- --ssl-certfile "$cert" --ssl-keyfile "$cert_key" "$@"
    fi
    if [ -n "$key" ]; then
        VLLM_API_KEY="$key"
        export VLLM_API_KEY
    fi
    exec vllm serve "$model" --served-model-name "$name" --host 0.0.0.0 --port "$port" \
        --max-num-seqs "$slots" --max-model-len "$context" "$@"
    ;;
*)
    echo "STOREYPATH_HELPER_ENGINE: llama.cpp or vllm, not $engine" >&2
    exit 2
    ;;
esac
