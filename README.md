# StoreyPath GPU helper

The vision model that [StoreyPath Studio](https://github.com/StoreyPath/storeypath-studio)
asks about the plans, served on a machine with an NVIDIA GPU. Studio reads a drawing
by rules and a small language model on its own; with a helper it also shows each
room's picture to a vision model (Gemma 4 31B) for the judgement calls: what a room is,
whether a door is a door. The helper is optional, and separate from Studio: one Studio
can use several helpers, and several Studios one.

It comes as two images, one engine each, with the same contract: an OpenAI-compatible
API on port 8105, every call but `/health` needing a key, HTTPS by default, and the
model baked in. Nothing is downloaded when it runs, so it works on a machine with no
network. What the key and HTTPS protect, and what they don't: [Security](#security).

## Which image

| | **vLLM** (recommended): `Dockerfile.vllm` | **llama.cpp**: `Dockerfile` |
|---|---|---|
| Model | Gemma 4 31B, FP8 (8-bit), served as `gemma-4-31B-it-FP8` | Gemma 4 31B, 4-bit (Q4_K_M), served as `gemma-4-31B-it-Q4_K_M` |
| NVIDIA driver | 580 or newer (CUDA 13.0); a B300 580.82 or newer; on a data-center GPU (A100, H100, H200, B200, L40S…) also the 535 and 570 branches, through the CUDA forward-compatibility libraries inside the image. Each GPU: the table below | 525 or newer |
| GPU | compute capability 8.0 or newer: A100, H100, H200, B200, B300, RTX PRO 6000 Blackwell and others. FP8 runs natively from 8.9 (L4, L40S, H100, Blackwell); an A100 reads the 8-bit weights through slower kernels | compute capability 8.0 or newer |
| GPU memory | 44 GB by default (`STOREYPATH_HELPER_GPU_MEMORY`), about 46 GB used in all | about 32 GB with 2 questions at once |
| Questions at once | 16 (`STOREYPATH_HELPER_SLOTS`) | 2 |
| Start | about 3 minutes (it compiles for the GPU each time) | a minute or two |
| Image | about 37 GB to move (`docker save … \| gzip`), about 56 GB once loaded (as `docker images` counts it) | about 21 GB |

The vLLM image is built on vLLM 0.31.0 for CUDA 13.0, which has kernels for each of
these GPUs. What each one needs and takes, from the kernels inside the image and vLLM's
own choices for Gemma 4 on it:

| GPU | Compute capability | NVIDIA driver | FP8 | Attention |
|---|---|---|---|---|
| A100 | 8.0 | 580 or newer, or 535 or 570 | weight-only: the 8-bit weights are widened as each layer reads them (Humming's kernels, compiled for the GPU as the helper starts; Marlin's next in line), the arithmetic in BF16 | Triton |
| H100, H200 | 9.0 | 580 or newer, or 535 or 570 (an H200 535.129 or newer) | native FP8 (CUTLASS, Hopper's kernels) | Triton for the steps that read a picture, FlashAttention 4 for the rest |
| B200 | 10.0 | 580 or newer, or 570 | native FP8 (CUTLASS, Blackwell's family kernels, sm_100f) | Triton for the steps that read a picture, FlashInfer (TensorRT-LLM's kernels) for the rest |
| B300 | 10.3 | 580.82 or newer (no older driver knows it) | as a B200: the same family kernels run on it | as a B200 (FlashInfer has kernels of its own for it) |
| RTX PRO 6000 Blackwell | 12.0 | 580 or newer (an RTX PRO card has no forward compatibility) | native FP8 (CUTLASS, sm_120f) | Triton |
| Others of compute capability 8.0 or newer with about 46 GB (L40S, A40, RTX 6000 Ada…) | 8.6, 8.9 | 580 or newer; a data-center GPU also 535 or 570 | native from 8.9, weight-only below | Triton |

The vision tower (the part of Gemma that looks at the picture) runs on PyTorch's own
attention on every GPU. The first start on a GPU also compiles kernels for it (Triton's,
FlashAttention 4's, Humming's) and so takes longer. [smoke-test.sh](#smoke-test) tries an
image on a GPU. On an RTX PRO or GeForce card whose driver is older than 580, build the
image on vLLM's CUDA 12.9 build instead ([Build](#build-on-a-connected-machine)): driver
575 or newer, every GPU above but a B300.

Both read rooms as well as each other; vLLM reads them two to four times faster. Take
the llama.cpp image where the driver is older than vLLM's CUDA needs, or where the GPU
has less than about 46 GB free.

Measured on one NVIDIA RTX PRO 6000 Blackwell (96 GB, the 300 W Max-Q card), on the same
rooms, with Studio's own question (its picture of a room, the choices, a JSON answer):

| Engine and model | Right | Rooms a minute, one at a time | Rooms a minute, 16 at once |
|---|---|---|---|
| vLLM, Gemma 4 31B FP8 | 81% | 53–55 | 87–91 |
| vLLM, Qwen 3.5 27B FP8 (`MODEL=Qwen3.5-27B-FP8`, below) | 82% | 83 | 316–323 |
| llama.cpp, Gemma 4 31B 4-bit | 82% | about 22 | about 22 (whatever the parallelism) |

## Build (on a connected machine)

```sh
# vLLM
./fetch-vision.sh fp8                                     # once: about 33 GB, checksum-verified, into models/vllm/
docker build --platform linux/amd64 -f Dockerfile.vllm -t storeypath/gpu-helper:vllm .
docker save storeypath/gpu-helper:vllm | gzip -1 > storeypath-gpu-helper-vllm.tar.gz

# llama.cpp
./fetch-vision.sh                                         # once: about 19 GB, checksum-verified, into models/vision/
docker build --platform linux/amd64 -t storeypath/gpu-helper .
docker save storeypath/gpu-helper | gzip -1 > storeypath-gpu-helper.tar.gz
```

The images are large (the model is inside), so they are built here, not by CI. Each
Dockerfile has its own build context (`.dockerignore`, `Dockerfile.vllm.dockerignore`):
only its model is sent to Docker. Building the vLLM image takes a quarter of an hour,
most of it compressing the model's layer.

For an RTX PRO or GeForce card whose NVIDIA driver is 575 to 579 (they have no forward
compatibility, and the image's CUDA 13.0 needs 580), build on vLLM's CUDA 12.9 image
instead. It serves every GPU in the table above but a B300, whose FP8 kernels it lacks:

```sh
docker build --platform linux/amd64 -f Dockerfile.vllm -t storeypath/gpu-helper:vllm-cu129 \
    --build-arg VLLM_IMAGE=vllm/vllm-openai:v0.31.0-cu129@sha256:ff29e51a9457f191deb332b96fa6440584fa941aa2a548449b604e2b1e5e1eeb .
```

The vLLM image can carry Qwen 3.5 27B (FP8) instead of Gemma: `./fetch-vision.sh
qwen-fp8`, then build with `--build-arg MODEL=Qwen3.5-27B-FP8 --build-arg
IMAGE_KWARGS={}` (and e.g. `-t storeypath/gpu-helper:vllm-qwen`). It is served as
`Qwen3.5-27B-FP8`; Studio files answers by the model's name, so it asks every room
again.

## Run (on the GPU machine)

Linux with Docker and the NVIDIA Container Toolkit, and the driver the table above asks
for.

```sh
openssl rand -hex 32 > helper.key                         # the key: at least 32 characters
printf 'STOREYPATH_HELPER_KEY=%s\n' "$(cat helper.key)" > helper.env && chmod 600 helper.env helper.key

# vLLM
docker load -i storeypath-gpu-helper-vllm.tar.gz
docker run -d --name storeypath-gpu --gpus '"device=0"' -p 8105:8105 --env-file helper.env storeypath/gpu-helper:vllm

# llama.cpp
docker load -i storeypath-gpu-helper.tar.gz
docker run -d --name storeypath-gpu --gpus '"device=0"' -p 8105:8105 --env-file helper.env storeypath/gpu-helper
```

It speaks HTTPS with a certificate it makes as it starts; `docker logs storeypath-gpu`
shows its SHA-256 fingerprint. (`--env-file` keeps the key off the command line, which
others on the machine can see.)

Choose the card with `--gpus '"device=1"'` (one card: the helper uses the first it
sees). Until the model is loaded `/health` does not answer 200 (llama.cpp: 503; vLLM:
nothing listens yet); Studio leaves the helper out until it does.

On a data-center GPU whose driver is older than the image's CUDA (the 535 or 570
branch), the NVIDIA Container Toolkit puts the image's forward-compatibility libraries to
use. If yours does not (an older toolkit: the helper stops with an error that the driver
is too old for its CUDA), add `-e VLLM_ENABLE_CUDA_COMPATIBILITY=1` and vLLM puts them to
use itself.

### Smoke test

[smoke-test.sh](smoke-test.sh) tries an image on one GPU before Studio uses it: it starts
the image with no network (`docker run --network none`), waits for `/health`, checks that
a call without the key is refused, asks one question about a made-up plan through the
OpenAI API with the key, and checks that the answer comes back as JSON naming the word
written in it. It prints the GPU, what vLLM chose for it (attention, FP8 kernels), the
answer and how long each step took, and removes its container.

```sh
./smoke-test.sh storeypath/gpu-helper:vllm 0                     # the GPU nvidia-smi numbers 0
./smoke-test.sh storeypath/gpu-helper:vllm nvidia.com/gpu=0      # with the toolkit's CDI
STOREYPATH_HELPER_GPU_MEMORY=auto ./smoke-test.sh storeypath/gpu-helper:vllm 1   # a GPU others use too
./smoke-test.sh storeypath/gpu-helper:vllm 0 --linear-backend marlin   # arguments after the GPU go to vLLM
```

It makes a key of its own for the test unless `STOREYPATH_HELPER_KEY` is set, and never
prints it. The first start on a GPU compiles kernels for it; it waits up to half an hour
(`SMOKE_TIMEOUT`, in seconds).

On a GPU other programs share, the vLLM helper takes the memory it is given
(`STOREYPATH_HELPER_GPU_MEMORY`) and about 2 GB more; that much must be free when it
starts, or it stops with a message saying how much is. Measured with Studio's question:

| `STOREYPATH_HELPER_GPU_MEMORY` | GPU memory used (nvidia-smi) | Questions at once | Rooms a minute, 16 sent at once |
|---|---|---|---|
| 40 | does not start: one question of 8192 tokens does not fit | | |
| 40, with `STOREYPATH_HELPER_CONTEXT=4096` | 41.1 GB | 5–6 | 85 |
| **44** (default) | 45.7 GB | about 10 (the rest wait their turn) | 87–89 |
| 48 | 49.6 GB | 13 | 91 |
| 50 | 51.6 GB | 15–16 | 90 |
| 52 | 53.6 GB | 16 | 90 |
| `auto` | what is free on the card when it starts, less 2 GB, up to 52 (52.0 → 51.9 GB used on an empty card) | up to 16 | as above |

On that card the GPU itself was the limit (busy all the time, at its power cap), so more
questions at once did not read rooms faster; on a faster card more may. A room's question
is about 1,450 tokens (Gemma sees each picture in up to 1,120), so a context of 4096
is enough for it where memory is short. The Qwen build at the default 44 GB holds all 16
at once (its memory per question is far smaller) and used 45.5 GB.

## Studio

Tell Studio where the helper is: on Studio's *GPU helpers* page (admins), or when
starting Studio:

| Studio's variable | |
|---|---|
| `STOREYPATH_VISION_URL` | `https://<gpu machine>:8105/v1` (`http://` with `STOREYPATH_HELPER_PLAIN_HTTP=1`); several helpers: their addresses separated by commas, and Studio spreads a floor's rooms over them |
| `STOREYPATH_VISION_KEY` | the helper's `STOREYPATH_HELPER_KEY` |
| `STOREYPATH_VISION_PARALLEL` | the helper's slots: 16 for vLLM, 2 for llama.cpp |
| `STOREYPATH_VISION_MODEL` | the served name: `gemma-4-31B-it-FP8` (vLLM), `gemma-4-31B-it-Q4_K_M` (llama.cpp). Studio files answers by it: rooms read by one model are asked again of another |
| `STOREYPATH_VISION_CA` | optional: your organisation's CA certificate (PEM), when the helper has a certificate from it. Studio then checks the helper's certificate and refuses one it does not vouch for ([Security](#security)) |

Studio's `docker/compose.yml` runs Studio and a helper on one machine.

## Security

**What is protected.** The helper refuses every call without the key, except `/health`.
The key must be at least 32 characters (`openssl rand -hex 32` makes one); the helper
does not start with a shorter one, or with none (unless `STOREYPATH_HELPER_OPEN=1`).
By default the helper speaks HTTPS, so the traffic between Studio and the helper,
the key included, is encrypted: someone watching the network sees neither the key
nor the plans.

**What isn't.** With the helper's own certificate (the one it makes when given none),
Studio does not check *who* it is talking to. Someone on your network who can pose as
the helper (take its address or name) could receive Studio's calls and collect the key.
Studio says so: on its GPU helpers page and in its log the helper shows as *HTTPS, its
own certificate (not checked)*. To close that, either:

- give the helper a certificate from your organisation's CA (`STOREYPATH_HELPER_CERT`,
  `STOREYPATH_HELPER_CERT_KEY`, for the name Studio uses to reach it) and give Studio
  that CA (`STOREYPATH_VISION_CA`). Studio then checks the certificate, refuses a helper
  whose certificate the CA does not vouch for, and shows *HTTPS, certificate checked*; or
- keep the helpers on a network, or behind a firewall, that only Studio can reach.

**Plain HTTP.** `STOREYPATH_HELPER_PLAIN_HTTP=1` turns HTTPS off: then the calls and the
key cross the network as they are. Use it only where something else encrypts them, such
as an SSH tunnel or a proxy in front that speaks HTTPS.

**Rotating the key.** Make a new one, restart the helper with it (`docker rm -f` and
`docker run` again with the new `--env-file`), and give Studio the same new key (its
GPU helpers page, or `STOREYPATH_VISION_KEY`). Calls with the old key are refused from
then on.

**`/health`** answers without the key, so that Studio and Docker's health check can ask
it. It says only whether the model is loaded: 200 with an empty reply (vLLM) or
`{"status":"ok"}` (llama.cpp), else an error or no answer while it loads. Nothing about
the model, the key or the plans (vLLM's reply does name its web server, uvicorn, in a
header).

## Settings

| Variable | |
|---|---|
| `STOREYPATH_HELPER_KEY` | the key every call but `/health` must carry (Studio's `STOREYPATH_VISION_KEY`), at least 32 characters (`openssl rand -hex 32`). Without one, or with a shorter one, the helper does not start, unless `STOREYPATH_HELPER_OPEN=1` (a network nothing else can reach) |
| `STOREYPATH_HELPER_SLOTS` | questions answered at once (vLLM 16, llama.cpp 2; with llama.cpp each one more needs more GPU memory). Studio's `STOREYPATH_VISION_PARALLEL` should match it |
| `STOREYPATH_HELPER_GPU_MEMORY` | vLLM: the GPU memory it is given, in GB (`44`), as a share of the card (`0.5`), or `auto` (what is free when it starts, less 2 GB, up to 52: for a GPU others use too; it refuses to start with less than about 41); default 44 GB, and it uses about 2 GB more. See the table above |
| `STOREYPATH_HELPER_CONTEXT` | tokens of context per question (default 8192) |
| `STOREYPATH_HELPER_CERT`, `STOREYPATH_HELPER_CERT_KEY` | optional: HTTPS with this certificate and key (PEM, mounted into the container, readable by its user, uid 10001), e.g. from your organisation's CA, which Studio then checks (`STOREYPATH_VISION_CA`). Without them the helper makes its own as it starts (self-signed, EC P-256, ten years; its SHA-256 fingerprint in the log), which encrypts but which Studio cannot check ([Security](#security)) |
| `STOREYPATH_HELPER_PLAIN_HTTP` | `1`: plain HTTP, no certificate: for an SSH tunnel, or a proxy in front that speaks HTTPS ([Security](#security)) |
| `STOREYPATH_HELPER_NAME` | the name the model is served under (default `gemma-4-31B-it-FP8` for vLLM, `gemma-4-31B-it-Q4_K_M` for llama.cpp). Studio files answers by it: helpers serving the same model use the same name |
| `STOREYPATH_HELPER_IMAGE_KWARGS` | vLLM: how the model's image processor reads a picture, as JSON; the image sets `{"max_soft_tokens":1120}` for Gemma (up to 1,120 tokens a picture, as measured; Gemma's own default, 280, is coarse for a plan) |
| `STOREYPATH_HELPER_ENGINE` | `vllm` or `llama.cpp`: each image sets its own |

Further arguments to `docker run … <image>` go to the engine (`vllm serve`,
`llama-server`) as they are. Details are in [start.sh](start.sh) and the two
Dockerfiles.

## Licence

Apache 2.0 ([LICENSE](LICENSE)). Inside the llama.cpp image: llama.cpp (MIT), OpenSSL
(Apache-2.0), NVIDIA's CUDA runtime and cuBLAS (CUDA Toolkit EULA, redistributable) and
Gemma 4's weights (Apache-2.0). Inside the vLLM image: vLLM (Apache-2.0), PyTorch
(BSD-3-Clause), the other Python packages of vLLM's image (each listed with its licence;
among them the GPU kernels it takes, FlashInfer with its prebuilt kernels and Humming,
Apache-2.0, and the NVIDIA CUTLASS DSL that compiles FlashAttention 4, under NVIDIA's
software licence), NVIDIA's CUDA libraries as vLLM's image ships them (NVIDIA Deep
Learning Container License) and the model's weights (Gemma 4 or Qwen 3.5: Apache-2.0).
Each image keeps its licences in `/usr/share/storeypath/licenses`.
