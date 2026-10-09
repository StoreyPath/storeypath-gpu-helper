#!/bin/sh
# StoreyPath's GPU helper: the vision model server, OpenAI-compatible, on port 8105.
# Two builds, one engine each: vLLM (Dockerfile.vllm: faster, a newer NVIDIA driver)
# and llama.cpp (Dockerfile: older drivers, less memory). Arguments are passed on to
# the engine (vllm serve, llama-server) as they are.
#
#   STOREYPATH_HELPER_KEY       the key every call but /health must carry (Studio sends
#                               it: its STOREYPATH_VISION_KEY), at least 32 characters
#                               (openssl rand -hex 32 makes one). Without one the helper
#                               does not start, unless STOREYPATH_HELPER_OPEN=1 (a
#                               network nothing else can reach)
#   STOREYPATH_HELPER_SLOTS     questions answered at once (default 16 with vLLM, 2
#                               with llama.cpp); Studio's STOREYPATH_VISION_PARALLEL
#                               should match it
#   STOREYPATH_HELPER_CONTEXT   tokens of context per question (default 8192)
#   STOREYPATH_HELPER_GPU_MEMORY  vLLM: the GPU memory it is given, in GB (e.g. 44) or
#                               as a share of the card (e.g. 0.5), or auto; default 44 GB,
#                               which holds about 10 room questions at once (the rest wait
#                               their turn), and it uses about 2 GB more than this. That
#                               much must be free when it starts. 52 GB holds all 16 at
#                               once; below about 41 it does not start with 8192 tokens of
#                               context. auto: what is free on the card when it starts,
#                               less 2 GB, up to 52 (for a GPU others use too). (llama.cpp
#                               takes what the model and its slots need: not read)
#   STOREYPATH_HELPER_CERT      HTTPS with this certificate (PEM, its chain after it)
#   STOREYPATH_HELPER_CERT_KEY  and its private key (PEM): e.g. one from your
#                               organisation's CA, which Studio can check
#                               (STOREYPATH_VISION_CA). Without them the helper still
#                               speaks HTTPS, with a self-signed certificate it makes as
#                               it starts (its SHA-256 fingerprint in the log): the
#                               traffic and the key are encrypted, but Studio cannot
#                               check who it is talking to (README.md, Security)
#   STOREYPATH_HELPER_PLAIN_HTTP  1: plain HTTP, no certificate (for an SSH tunnel, or
#                               a proxy in front that speaks HTTPS)
#   STOREYPATH_HELPER_NAME      the name the model is served under (default: the model
#                               file's or folder's, e.g. gemma-4-31B-it-Q4_K_M or
#                               gemma-4-31B-it-FP8). Studio files each answer by it:
#                               helpers serving the same model use the same name
#   STOREYPATH_HELPER_MODEL     the model: a GGUF file (llama.cpp) or a folder in
#                               Hugging Face's format (vLLM); the image sets it
#   STOREYPATH_HELPER_MMPROJ    llama.cpp: the model's image encoder (the image sets it)
#   STOREYPATH_HELPER_IMAGE_KWARGS  vLLM: how the model's image processor reads a
#                               picture, as JSON (the image sets {"max_soft_tokens":
#                               1120} for Gemma 4: up to 1120 tokens a picture, as
#                               llama.cpp's --image-min-tokens 1024; its own default,
#                               280, is coarse for a plan)
#   STOREYPATH_HELPER_ENGINE    llama.cpp (the default) or vllm: the image sets it
#
# It takes a minute or two to load (vLLM: three or four, compiling for the GPU): /health
# answers 200 once it has (llama.cpp: 503 while loading; vLLM: no answer), and Studio
# leaves the helper out until then.
set -eu
port=8105
engine="${STOREYPATH_HELPER_ENGINE:-llama.cpp}"
case "$engine" in
llama.cpp) default_slots=2 ;;
vllm) default_slots=16 ;;
*)
    echo "STOREYPATH_HELPER_ENGINE: llama.cpp or vllm, not $engine" >&2
    exit 2
    ;;
esac
slots="${STOREYPATH_HELPER_SLOTS:-$default_slots}"
context="${STOREYPATH_HELPER_CONTEXT:-8192}"
key="${STOREYPATH_HELPER_KEY:-}"
model="${STOREYPATH_HELPER_MODEL:?STOREYPATH_HELPER_MODEL: the model to serve}"
name="${STOREYPATH_HELPER_NAME:-$(basename "$model" .gguf)}"
cert="${STOREYPATH_HELPER_CERT:-}"
cert_key="${STOREYPATH_HELPER_CERT_KEY:-}"
unset STOREYPATH_HELPER_KEY

plain="${STOREYPATH_HELPER_PLAIN_HTTP:-}"

if [ "${STOREYPATH_HELPER_OPEN:-}" != 1 ]; then
    if [ -z "$key" ]; then
        echo "no STOREYPATH_HELPER_KEY: give the helper a key, the one Studio sends as STOREYPATH_VISION_KEY" >&2
        echo "(make one with: openssl rand -hex 32), or STOREYPATH_HELPER_OPEN=1 to serve without one" >&2
        exit 2
    fi
    if [ "${#key}" -lt 32 ]; then
        echo "STOREYPATH_HELPER_KEY is too short (${#key} characters): it must be at least 32" >&2
        echo "(make one with: openssl rand -hex 32, and give Studio the same as STOREYPATH_VISION_KEY)" >&2
        exit 2
    fi
fi
case "$plain" in
'' | 0) plain= ;;
1) ;;
*)
    echo "STOREYPATH_HELPER_PLAIN_HTTP: 1 (plain HTTP) or nothing (HTTPS), not $plain" >&2
    exit 2
    ;;
esac
if [ -n "$cert$cert_key" ]; then
    if [ -n "$plain" ]; then
        echo "STOREYPATH_HELPER_PLAIN_HTTP=1 and a certificate (STOREYPATH_HELPER_CERT): one or the other" >&2
        exit 2
    fi
    if [ ! -r "$cert" ] || [ ! -r "$cert_key" ]; then
        echo "STOREYPATH_HELPER_CERT and STOREYPATH_HELPER_CERT_KEY: both must name readable PEM files" >&2
        exit 2
    fi
elif [ -z "$plain" ]; then
    # no certificate given: one of its own, made now (self-signed, EC P-256, ten years),
    # in a folder only this user reads. It encrypts; Studio cannot check it (README.md)
    tls="$(umask 077 && mktemp -d "${TMPDIR:-/tmp}/storeypath-helper-tls.XXXXXX")" || {
        echo "no writable temporary folder (${TMPDIR:-/tmp}) for the helper's own certificate: give TMPDIR one," \
             "or STOREYPATH_HELPER_CERT and STOREYPATH_HELPER_CERT_KEY, or STOREYPATH_HELPER_PLAIN_HTTP=1" >&2
        exit 2
    }
    cert="$tls/cert.pem"
    cert_key="$tls/key.pem"
    if ! (umask 077 && openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -days 3650 \
            -subj "/CN=storeypath-gpu-helper" -keyout "$cert_key" -out "$cert" >/dev/null 2>&1); then
        echo "the helper's own certificate could not be made (openssl): give STOREYPATH_HELPER_CERT and" \
             "STOREYPATH_HELPER_CERT_KEY, or STOREYPATH_HELPER_PLAIN_HTTP=1" >&2
        exit 2
    fi
    fingerprint="$(openssl x509 -in "$cert" -noout -fingerprint -sha256 | sed 's/^.*=//')"
    echo "HTTPS with a certificate of its own, made now (self-signed, for ten years): SHA-256 $fingerprint." \
         "Calls and the key are encrypted, but Studio cannot check it is this helper it talks to: for that, give" \
         "the helper a certificate from your organisation's CA (STOREYPATH_HELPER_CERT, _CERT_KEY) and Studio" \
         "the CA (STOREYPATH_VISION_CA)"
fi
scheme=https
if [ -n "$plain" ]; then
    scheme=http
    echo "plain HTTP (STOREYPATH_HELPER_PLAIN_HTTP=1): calls and the key cross the network as they are; for an SSH" \
         "tunnel, or a proxy in front that speaks HTTPS"
fi
# for Docker's health check (the Dockerfiles): which scheme /health is asked on
echo "$scheme" > "${TMPDIR:-/tmp}/storeypath-helper.scheme" 2>/dev/null || true

# vLLM: STOREYPATH_HELPER_GPU_MEMORY (GB, or a share of the card) as the share of the
# card's memory vLLM is told it may take (--gpu-memory-utilization)
gpu_share() {
    want="$(echo "$1" | sed 's/[[:space:]]//g; s/[Gg][Ii]\{0,1\}[Bb]\{0,1\}$//')"
    case "$want" in
    [Aa][Uu][Tt][Oo])
        # what is free on the first GPU this container sees, less a margin, up to what
        # 16 questions at once need: for a GPU others use too
        free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -dc 0-9)"
        if [ -z "$free_mib" ]; then
            echo "STOREYPATH_HELPER_GPU_MEMORY=auto: the GPU's free memory cannot be read (nvidia-smi); give GB, e.g. 44" >&2
            return 2
        fi
        want="$(awk -v f="$free_mib" 'BEGIN { g = f / 1024 - 2; if (g > 52) g = 52; printf "%.1f\n", g }')"
        if awk -v w="$want" 'BEGIN { exit !(w < 41) }'; then
            echo "STOREYPATH_HELPER_GPU_MEMORY=auto: only $want GB free to use on this GPU; it needs about 41 (or a smaller STOREYPATH_HELPER_CONTEXT)" >&2
            return 2
        fi
        echo "GPU memory: auto, $want GB (of what is free now)" >&2
        ;;
    esac
    case "$want" in
    '' | *[!0-9.]* | *.*.* | .)
        echo "STOREYPATH_HELPER_GPU_MEMORY: GB (e.g. 44) or a share of the card (e.g. 0.5), not $1" >&2
        return 2
        ;;
    esac
    if awk -v w="$want" 'BEGIN { exit !(w <= 0) }'; then
        echo "STOREYPATH_HELPER_GPU_MEMORY: more than nothing, not $1" >&2
        return 2
    fi
    if awk -v w="$want" 'BEGIN { exit !(w <= 1) }'; then
        echo "$want"
        return 0
    fi
    # GB: of the first GPU this container sees (choose it with --gpus)
    total_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -dc 0-9)"
    if [ -z "$total_mib" ]; then
        echo "STOREYPATH_HELPER_GPU_MEMORY=$1: the GPU's memory cannot be read (nvidia-smi); give a share of the card instead, e.g. 0.5" >&2
        return 2
    fi
    awk -v w="$want" -v t="$total_mib" -v given="$1" 'BEGIN {
        s = w * 1024 / t
        if (s > 1) { printf "STOREYPATH_HELPER_GPU_MEMORY=%s: more than the GPU has (%.1f GB)\n", given, t / 1024 > "/dev/stderr"; exit 2 }
        printf "%.4f\n", s
    }'
}

case "$engine" in
llama.cpp)
    echo "serving $name ($engine) at $scheme://<this machine>:$port/v1: $slots question(s) at once, $context tokens each," \
         "$(if [ -n "$key" ]; then echo "a key needed"; else echo "OPEN: no key needed"; fi)"
    if [ "$scheme" = https ]; then
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
    memory="${STOREYPATH_HELPER_GPU_MEMORY:-44}"
    share="$(gpu_share "$memory")" || exit 2
    echo "serving $name ($engine) at $scheme://<this machine>:$port/v1: $slots question(s) at once, $context tokens each," \
         "GPU memory $memory (share $share)," \
         "$(if [ -n "$key" ]; then echo "a key needed"; else echo "OPEN: no key needed"; fi)"
    if [ "$scheme" = https ]; then
        set -- --ssl-certfile "$cert" --ssl-keyfile "$cert_key" "$@"
    fi
    if [ -n "${STOREYPATH_HELPER_IMAGE_KWARGS:-}" ] && [ "$STOREYPATH_HELPER_IMAGE_KWARGS" != "{}" ]; then
        set -- --mm-processor-kwargs "$STOREYPATH_HELPER_IMAGE_KWARGS" "$@"
    fi
    # the key from the environment, not the command line (which the host's ps shows):
    # vLLM's own check (on /v1/...) and storeypath_key_guard (on everything but /health)
    if [ -n "$key" ]; then
        VLLM_API_KEY="$key"
        export VLLM_API_KEY
        set -- --middleware storeypath_key_guard.KeyGuard "$@"
    fi
    exec vllm serve "$model" --served-model-name "$name" --host 0.0.0.0 --port "$port" \
        --max-num-seqs "$slots" --max-model-len "$context" --gpu-memory-utilization "$share" \
        --limit-mm-per-prompt '{"image": 1, "video": 0}' "$@"
    ;;
esac
