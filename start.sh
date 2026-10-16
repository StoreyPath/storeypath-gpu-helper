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
#   STOREYPATH_HELPER_CONTEXT   tokens of context per question (default 8192; vLLM's small
#                               copy given less than 24 GB: 4096)
#   STOREYPATH_HELPER_MODEL     the model. vLLM's image carries Gemma 4 31B twice: auto (its
#                               default) takes the full copy (FP8) where the GPU memory the
#                               helper may use holds it, 41 GB or more, else the small copy
#                               (4-bit, for a 24 or 32 GB card), from 21.5 GB; full or small
#                               takes that copy. The memory it may use: STOREYPATH_HELPER_GPU_
#                               MEMORY when given in GB or as a share, else what is free on
#                               the card when it starts, less what vLLM uses beyond what it is
#                               given (2 GB, the small copy 1). Or the model itself: a GGUF
#                               file (llama.cpp: the image sets it) or a folder in Hugging
#                               Face's format (vLLM)
#   STOREYPATH_HELPER_FULL, STOREYPATH_HELPER_SMALL  vLLM: the two copies' folders (the image
#                               sets them)
#   STOREYPATH_HELPER_GPU_MEMORY  vLLM: the GPU memory it is given, in GB (e.g. 44) or as a
#                               share of the card (e.g. 0.5), or auto. By default the full copy
#                               is given 44 GB, which holds about 10 room questions at once
#                               (the rest wait their turn), and it uses about 2 GB more than
#                               it is given; below 41 it does not start with 8192 tokens of
#                               context. The small copy is given what is free less 1 GB (it
#                               uses about 1 more), up to 31 (all 16 at once); 22 GB hold 3
#                               or 4, and below 21.5 it does not start. Less is given where
#                               less is free. auto: what is free on the card when it starts,
#                               less what it uses beyond, up to what 16 questions at once need
#                               (the full copy 52, the small 31): for a GPU others use too.
#                               (llama.cpp takes what the model and its slots need: not read)
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
#   STOREYPATH_HELPER_MMPROJ    llama.cpp: the model's image encoder (the image sets it)
#   STOREYPATH_HELPER_IMAGE_KWARGS  vLLM: how the model's image processor reads a
#                               picture, as JSON; for Gemma 4 {"max_soft_tokens":1120} by
#                               default: up to 1120 tokens a picture, as llama.cpp's
#                               --image-min-tokens 1024 (its own default, 280, is coarse for
#                               a plan); {} for the model's own
#   STOREYPATH_HELPER_ENGINE    llama.cpp (the default) or vllm: the image sets it
#
# It takes a minute or two to load (vLLM: two to four, compiling for the GPU): /health
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
context="${STOREYPATH_HELPER_CONTEXT:-}" # (the default: 8192, or the small copy's below)
key="${STOREYPATH_HELPER_KEY:-}"
model="${STOREYPATH_HELPER_MODEL:?STOREYPATH_HELPER_MODEL: the model to serve}"
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

# vLLM: which copy of the model it serves (STOREYPATH_HELPER_MODEL: auto, full or small)
# and the GPU memory it is given (STOREYPATH_HELPER_GPU_MEMORY), in GB as nvidia-smi
# counts them (1024 MiB), of the first GPU this container sees (choose it with --gpus).
# For each copy: the least it starts with (one question of its context), what it is given
# by default, and what all 16 questions at once need; and how much more than it is given
# it takes (kept free when what is free decides)
FULL_LEAST=41 FULL_USUAL=44 FULL_MOST=52 FULL_MORE=2
SMALL_LEAST=21.5 SMALL_MOST=31 SMALL_MORE=1
SMALL_LONG=24 # the small copy: 8192 tokens of context from this much, 4096 below

gpu_mib() { # memory.free or memory.total of the first GPU, in MiB; nothing when unreadable
    nvidia-smi --query-gpu="memory.$1" --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -dc 0-9
}
at_least() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a >= b) }'; }
least_of() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.1f\n", (a < b ? a : b) }'; }

# what STOREYPATH_HELPER_GPU_MEMORY says: sets $given (GB) and $share, or $share alone
# for a share of a card whose memory cannot be read; nothing for auto or none given
read_memory() {
    given='' share=''
    want="$(echo "$1" | sed 's/[[:space:]]//g; s/[Gg][Ii]\{0,1\}[Bb]\{0,1\}$//')"
    case "$want" in
    '' | [Aa][Uu][Tt][Oo]) return 0 ;;
    *[!0-9.]* | *.*.* | .)
        echo "STOREYPATH_HELPER_GPU_MEMORY: GB (e.g. 44), a share of the card (e.g. 0.5) or auto, not $1" >&2
        return 2
        ;;
    esac
    if awk -v w="$want" 'BEGIN { exit !(w <= 0) }'; then
        echo "STOREYPATH_HELPER_GPU_MEMORY: more than nothing, not $1" >&2
        return 2
    fi
    if awk -v w="$want" 'BEGIN { exit !(w <= 1) }'; then
        share="$want"
        if [ -n "$total_mib" ]; then
            given="$(awk -v w="$want" -v t="$total_mib" 'BEGIN { printf "%.1f\n", w * t / 1024 }')"
        fi
        return 0
    fi
    if [ -z "$total_mib" ]; then
        echo "STOREYPATH_HELPER_GPU_MEMORY=$1: the GPU's memory cannot be read (nvidia-smi); give a share of the card instead, e.g. 0.5" >&2
        return 2
    fi
    if awk -v w="$want" -v t="$total_mib" 'BEGIN { exit !(w * 1024 > t) }'; then
        awk -v t="$total_mib" -v g="$1" 'BEGIN { printf "STOREYPATH_HELPER_GPU_MEMORY=%s: more than the GPU has (%.1f GB)\n", g, t / 1024 }' >&2
        return 2
    fi
    given="$want"
}

# the copy ($copy: full or small, nothing for a model of its own), its folder ($model),
# the memory it is given ($given GB, $share of the card) and its context
choose_vllm() {
    memory="${STOREYPATH_HELPER_GPU_MEMORY:-}"
    total_mib="$(gpu_mib total)"
    free_mib="$(gpu_mib free)"
    read_memory "$memory" || return 2
    case "$model" in
    auto)
        if [ -z "${STOREYPATH_HELPER_FULL:-}" ]; then
            echo "STOREYPATH_HELPER_MODEL=auto: STOREYPATH_HELPER_FULL names no model (the image sets it)" >&2
            return 2
        fi
        # what the helper may use: as given, else what is free less what vLLM takes beyond it
        if [ -n "$given" ]; then
            room_full="$given" room_small="$given"
        elif [ -n "$free_mib" ]; then
            room_full="$(awk -v f="$free_mib" -v m="$FULL_MORE" 'BEGIN { printf "%.1f\n", f / 1024 - m }')"
            room_small="$(awk -v f="$free_mib" -v m="$SMALL_MORE" 'BEGIN { printf "%.1f\n", f / 1024 - m }')"
        else
            room_full='' room_small=''
        fi
        if [ -z "$room_full" ] || at_least "$room_full" "$FULL_LEAST" || [ -z "${STOREYPATH_HELPER_SMALL:-}" ]; then
            copy=full
        else
            copy=small
        fi
        why="$(if [ -n "$given" ]; then echo "$given GB given"; elif [ -n "$room_full" ]; then
            echo "$(awk -v f="$free_mib" 'BEGIN { printf "%.1f", f / 1024 }') GB free on the GPU"; else
            echo "the GPU's free memory unknown"; fi)"
        ;;
    full | small)
        copy="$model" why="STOREYPATH_HELPER_MODEL=$model"
        room_full='' room_small=''
        if [ -z "$given" ] && [ -n "$free_mib" ]; then
            room_full="$(awk -v f="$free_mib" -v m="$FULL_MORE" 'BEGIN { printf "%.1f\n", f / 1024 - m }')"
            room_small="$(awk -v f="$free_mib" -v m="$SMALL_MORE" 'BEGIN { printf "%.1f\n", f / 1024 - m }')"
        fi
        ;;
    *)
        copy='' why=''
        room_full='' room_small=''
        if [ -z "$given" ] && [ -n "$free_mib" ]; then
            room_full="$(awk -v f="$free_mib" -v m="$FULL_MORE" 'BEGIN { printf "%.1f\n", f / 1024 - m }')"
        fi
        ;;
    esac
    case "$copy" in
    full) model="${STOREYPATH_HELPER_FULL:?STOREYPATH_HELPER_FULL: the folder of the full copy (the image sets it)}" ;;
    small) model="${STOREYPATH_HELPER_SMALL:?STOREYPATH_HELPER_SMALL: the folder of the small copy (the image sets it)}" ;;
    esac
    if [ "$copy" = small ]; then
        least="$SMALL_LEAST" usual="$SMALL_MOST" most="$SMALL_MOST" room="$room_small"
    else
        least="$FULL_LEAST" usual="$FULL_USUAL" most="$FULL_MOST" room="$room_full"
    fi
    case "$memory" in [Aa][Uu][Tt][Oo]) usual="$most" ;; esac
    if [ -z "$given" ] && [ -z "$share" ]; then
        # not given: what is usual for the copy, or what is free (less what it takes beyond) where less
        if [ -n "$room" ]; then
            given="$(least_of "$room" "$usual")"
            if ! at_least "$given" "$least"; then
                echo "only $(awk -v f="$free_mib" 'BEGIN { printf "%.1f", f / 1024 }') GB free on this GPU:" \
                     "the helper needs about $(awk -v l="$FULL_LEAST" -v m="$FULL_MORE" 'BEGIN { print l + m }') GB free" \
                     "for the full copy of the model, $(awk -v l="$SMALL_LEAST" -v m="$SMALL_MORE" 'BEGIN { print l + m }') for the small one" \
                     "(or give STOREYPATH_HELPER_GPU_MEMORY with a smaller STOREYPATH_HELPER_CONTEXT)" >&2
                return 2
            fi
        elif [ "$memory" != "" ]; then
            echo "STOREYPATH_HELPER_GPU_MEMORY=auto: the GPU's free memory cannot be read (nvidia-smi); give GB, e.g. 44" >&2
            return 2
        elif [ -n "$total_mib" ]; then
            given="$usual"
        else
            echo "the GPU's memory cannot be read (nvidia-smi): give STOREYPATH_HELPER_GPU_MEMORY as a share of the card, e.g. 0.5" >&2
            return 2
        fi
    fi
    if [ -z "$share" ]; then
        share="$(awk -v g="$given" -v t="$total_mib" 'BEGIN { printf "%.4f\n", g * 1024 / t }')"
    fi
    if [ -z "$context" ]; then
        context=8192
        if [ "$copy" = small ] && [ -n "$given" ] && ! at_least "$given" "$SMALL_LONG"; then
            context=4096
        fi
    fi
}

if [ "$engine" = vllm ]; then
    choose_vllm || exit 2
else
    context="${context:-8192}"
fi
name="${STOREYPATH_HELPER_NAME:-$(basename "$model" .gguf)}"

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
    echo "serving $name ($engine$(if [ -n "$copy" ]; then echo ", the $copy copy of the model: $why"; fi))" \
         "at $scheme://<this machine>:$port/v1: $slots question(s) at once, $context tokens each," \
         "GPU memory ${given:+$given GB, }share $share," \
         "$(if [ -n "$key" ]; then echo "a key needed"; else echo "OPEN: no key needed"; fi)"
    if [ "$scheme" = https ]; then
        set -- --ssl-certfile "$cert" --ssl-keyfile "$cert_key" "$@"
    fi
    # how a picture is read: as given, else Gemma 4 with up to 1120 tokens a picture
    kwargs="${STOREYPATH_HELPER_IMAGE_KWARGS-}"
    if [ -z "${STOREYPATH_HELPER_IMAGE_KWARGS+given}" ] && grep -q '"model_type": *"gemma4"' "$model/config.json" 2>/dev/null; then
        kwargs='{"max_soft_tokens":1120}'
    fi
    if [ -n "$kwargs" ] && [ "$kwargs" != "{}" ]; then
        set -- --mm-processor-kwargs "$kwargs" "$@"
    fi
    # the small copy, made for a 24 or 32 GB card: its memory for the questions (the KV
    # cache) in 8 bits, twice the questions at once, and at most 2048 tokens read in one
    # step (a picture and its question are about 1450), which needs less memory to start
    # (arguments given to the helper come after these, and win)
    if [ "$copy" = small ]; then
        set -- --kv-cache-dtype fp8 --max-num-batched-tokens 2048 "$@"
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
