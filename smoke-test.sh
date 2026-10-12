#!/bin/sh
# A smoke test of the vLLM helper image on one GPU, with no network: it starts the image
# (docker run --network none), waits for /health, asks one question about a picture
# through the OpenAI API with the key, checks that the answer comes back as JSON, and
# says how long each took. With no network nothing outside can reach the container, so
# the calls are made inside it, on the HTTPS the helper serves with its own certificate.
#
#   ./smoke-test.sh <image> <gpu> [engine arguments]
#
#   <image>  e.g. storeypath/gpu-helper:vllm
#   <gpu>    the GPU's index as nvidia-smi numbers it (docker run --gpus device=<gpu>), or
#            a CDI name such as nvidia.com/gpu=0 (docker run --device)
#   further arguments go to the engine as the helper passes them (vllm serve), e.g.
#            --linear-backend marlin to try an A100 on Marlin's FP8 kernels
#
# The key: STOREYPATH_HELPER_KEY from the environment (at least 32 characters), else one
# made for the test. It reaches the container by its name only, never on a command line,
# and is never printed. STOREYPATH_HELPER_GPU_MEMORY, _CONTEXT and _SLOTS go to the helper
# when set (e.g. STOREYPATH_HELPER_GPU_MEMORY=auto on a GPU others use). SMOKE_TIMEOUT:
# seconds to wait for /health (default 1800: a first start compiles kernels for the GPU).
# SMOKE_KEEP=1 leaves the container running afterwards (its name is printed).
#
# It prints the GPU, what vLLM chose for it (attention backends, the FP8 kernel) and the
# answer, and exits 0 when the answer is JSON naming the word written in the picture.
set -eu

if [ $# -lt 2 ]; then
    echo "usage: $0 <image> <gpu index or CDI name> [engine arguments], e.g. $0 storeypath/gpu-helper:vllm 0" >&2
    exit 2
fi
image="$1"
gpu="$2"
shift 2 # what is left: the engine's arguments
timeout="${SMOKE_TIMEOUT:-1800}"

if [ -z "${STOREYPATH_HELPER_KEY:-}" ]; then
    STOREYPATH_HELPER_KEY="$(openssl rand -hex 32 2>/dev/null || od -An -tx1 -N32 /dev/urandom | tr -d ' \n')"
fi
if [ "${#STOREYPATH_HELPER_KEY}" -lt 32 ]; then
    echo "STOREYPATH_HELPER_KEY is too short (${#STOREYPATH_HELPER_KEY} characters): the helper needs at least 32" >&2
    exit 2
fi
export STOREYPATH_HELPER_KEY

case "$gpu" in
*[!A-Za-z0-9./=_-]* | '')
    echo "<gpu>: an index (e.g. 0) or a CDI name (e.g. nvidia.com/gpu=0), not $gpu" >&2
    exit 2
    ;;
*=* | */*) gpu_flag=--device gpu_value="$gpu" ;;
*[!0-9]*)
    echo "<gpu>: an index (e.g. 0) or a CDI name (e.g. nvidia.com/gpu=0), not $gpu" >&2
    exit 2
    ;;
*) gpu_flag=--gpus gpu_value="device=$gpu" ;;
esac
# the helper's settings that are set here, passed by name
settings=
for v in STOREYPATH_HELPER_GPU_MEMORY STOREYPATH_HELPER_CONTEXT STOREYPATH_HELPER_SLOTS; do
    if eval "[ -n \"\${$v:-}\" ]"; then
        settings="$settings -e $v"
    fi
done

name="storeypath-smoke-$(echo "$gpu" | tr -c 'A-Za-z0-9\n' '-')-$$"
cleanup() {
    if [ "${SMOKE_KEEP:-}" = 1 ]; then
        echo "left running: $name (docker rm -f $name)"
    else
        docker rm -f "$name" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT
trap 'exit 1' INT TERM

started="$(date +%s)"
echo "starting $image on GPU $gpu, no network, as $name"
# shellcheck disable=SC2086 # $settings: -e and variable names, split as words
docker run -d --name "$name" --network none "$gpu_flag" "$gpu_value" $settings -e STOREYPATH_HELPER_KEY \
    "$image" "$@" >/dev/null

# /health: nothing answers while the model loads and compiles, then 200
while :; do
    if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]; then
        echo "FAIL: the helper stopped before it was ready; its log ends:" >&2
        docker logs --tail 40 "$name" >&2 2>&1 || true
        exit 1
    fi
    if docker exec "$name" curl -fsk -m 4 -o /dev/null https://127.0.0.1:8105/health 2>/dev/null; then
        break
    fi
    if [ $(($(date +%s) - started)) -ge "$timeout" ]; then
        echo "FAIL: /health did not answer within $timeout s; the log ends:" >&2
        docker logs --tail 40 "$name" >&2 2>&1 || true
        exit 1
    fi
    sleep 5
done
ready="$(date +%s)"
echo "ready after $((ready - started)) s"
gpu_info="$(docker exec "$name" nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null)" \
    || gpu_info="unknown (no nvidia-smi in the container)"
echo "GPU (name, compute capability, driver): $gpu_info"
echo "what vLLM chose here (from its log):"
docker logs "$name" 2>&1 | grep -iE "attention backend|vit attention|heterogeneous head|FlashAttention|Selected [A-Za-z0-9]+Kernel|weight-only|marlin|humming|cutlass" \
    | sed -E 's/^.*\] //' | cut -c1-200 | sort -u | sed 's/^/  /' || true

# the question, from inside the container: the key from its environment, which docker
# run gave it by name (the helper's own process drops it; docker exec still has it)
docker exec -i "$name" python3 - <<'EOF'
import base64, io, json, os, ssl, sys, time, urllib.error, urllib.request

from PIL import Image, ImageDraw, ImageFont

base = "https://127.0.0.1:8105"
key = os.environ["STOREYPATH_HELPER_KEY"]
ctx = ssl._create_unverified_context()  # its own certificate, on this container's loopback


def call(path, body=None, with_key=True):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if with_key:
        req.add_header("Authorization", "Bearer " + key)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=600) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, None


status, _ = call("/v1/models", with_key=False)
if status != 401:
    sys.exit(f"FAIL: /v1/models without the key answered {status}, not 401")
status, models = call("/v1/models")
if status != 200:
    sys.exit(f"FAIL: /v1/models with the key answered {status}")
model = models["data"][0]["id"]
print(f"served as {model}; without the key: 401, as it should")

# a made-up plan: one room, a door in its wall, a word written in it
img = Image.new("RGB", (768, 576), "white")
d = ImageDraw.Draw(img)
d.rectangle([64, 64, 704, 512], outline="black", width=10)
d.rectangle([330, 502, 440, 522], fill="white")
d.arc([330, 412, 530, 612], 180, 270, fill="black", width=3)
d.text((384, 270), "KITCHEN", fill="black", anchor="mm", font=ImageFont.load_default(size=80))
buf = io.BytesIO()
img.save(buf, format="PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

question = {
    "model": model,
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "This is a floor plan of one room. Which word is written inside the room? "
                                 "Answer with JSON only, like {\"word\": \"...\"}."}]}],
    "max_tokens": 64,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}
t = time.time()
status, answer = call("/v1/chat/completions", question)
took = time.time() - t
if status != 200:
    sys.exit(f"FAIL: the question answered {status}")
text = answer["choices"][0]["message"]["content"] or ""
print(f"answer in {took:.1f} s ({answer['usage']['prompt_tokens']} tokens in, "
      f"{answer['usage']['completion_tokens']} out): {text.strip()}")
body = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
try:
    word = str(json.loads(body)["word"])
except (ValueError, KeyError, TypeError):
    sys.exit("FAIL: the answer is not JSON with a word")
if "KITCHEN" not in word.upper():
    sys.exit(f"FAIL: the answer is JSON but names {word!r}, not the word in the picture (KITCHEN)")
print("PASS: a JSON answer naming the word in the picture")
EOF
echo "in all $(($(date +%s) - started)) s from docker run to the answer"
