# StoreyPath GPU helper

The vision model that [StoreyPath Studio](https://github.com/StoreyPath/storeypath-studio)
asks about the plans, served on a machine with an NVIDIA GPU. Studio reads a drawing
by rules and a small language model on its own; with a helper it also shows each
room's picture to a vision model (Gemma 4 31B) for the judgement calls: what a room is,
whether a door is a door. The helper is optional, and separate from Studio: one Studio
can use several helpers, and several Studios one.

It comes as two images, one engine each, with the same contract: an OpenAI-compatible
API on port 8105, every call but `/health` needing a key, HTTPS when given a
certificate, and the model baked in. Nothing is downloaded when it runs, so it works on
a machine with no network.

## Which image

| | **vLLM** (recommended): `Dockerfile.vllm` | **llama.cpp**: `Dockerfile` |
|---|---|---|
| Model | Gemma 4 31B, FP8 (8-bit), served as `gemma-4-31B-it-FP8` | Gemma 4 31B, 4-bit (Q4_K_M), served as `gemma-4-31B-it-Q4_K_M` |
| NVIDIA driver | 575 or newer (CUDA 12.9); on a data-center GPU (A100, H100, L40S…) also the 535, 550, 560, 565 and 570 branches, through the CUDA forward-compatibility libraries inside the image | 525 or newer |
| GPU | compute capability 8.0 or newer (A100 and newer). FP8 runs natively from 8.9 (L4, L40S, H100, Blackwell); an A100 reads the 8-bit weights through slower kernels | compute capability 8.0 or newer |
| GPU memory | 44 GB by default (`STOREYPATH_HELPER_GPU_MEMORY`), about 46 GB used in all | about 32 GB with 2 questions at once |
| Questions at once | 16 (`STOREYPATH_HELPER_SLOTS`) | 2 |
| Start | about 3 minutes (it compiles for the GPU each time) | a minute or two |
| Image | about 38 GB to move (`docker save … \| gzip`), about 71 GB once loaded | about 21 GB |

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

The vLLM image can carry Qwen 3.5 27B (FP8) instead of Gemma: `./fetch-vision.sh
qwen-fp8`, then build with `--build-arg MODEL=Qwen3.5-27B-FP8 --build-arg
IMAGE_KWARGS={}` (and e.g. `-t storeypath/gpu-helper:vllm-qwen`). It is served as
`Qwen3.5-27B-FP8`; Studio files answers by the model's name, so it asks every room
again.

## Run (on the GPU machine)

Linux with Docker and the NVIDIA Container Toolkit, and the driver the table above asks
for.

```sh
# vLLM
docker load -i storeypath-gpu-helper-vllm.tar.gz
docker run -d --name storeypath-gpu --gpus '"device=0"' -p 8105:8105 \
    -e STOREYPATH_HELPER_KEY=<a long random key, e.g. from openssl rand -hex 32> storeypath/gpu-helper:vllm

# llama.cpp
docker load -i storeypath-gpu-helper.tar.gz
docker run -d --name storeypath-gpu --gpus '"device=0"' -p 8105:8105 \
    -e STOREYPATH_HELPER_KEY=<a long random key> storeypath/gpu-helper
```

Choose the card with `--gpus '"device=1"'` (one card: the helper uses the first it
sees). Until the model is loaded `/health` does not answer 200 (llama.cpp: 503; vLLM:
nothing listens yet); Studio leaves the helper out until it does.

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
| `STOREYPATH_VISION_URL` | `http://<gpu machine>:8105/v1` (`https://` with a certificate); several helpers: their addresses separated by commas, and Studio spreads a floor's rooms over them |
| `STOREYPATH_VISION_KEY` | the helper's `STOREYPATH_HELPER_KEY` |
| `STOREYPATH_VISION_PARALLEL` | the helper's slots: 16 for vLLM, 2 for llama.cpp |
| `STOREYPATH_VISION_MODEL` | the served name: `gemma-4-31B-it-FP8` (vLLM), `gemma-4-31B-it-Q4_K_M` (llama.cpp). Studio files answers by it: rooms read by one model are asked again of another |

Studio's `docker/compose.yml` runs Studio and a helper on one machine.

## Settings

| Variable | |
|---|---|
| `STOREYPATH_HELPER_KEY` | the key every call must carry (Studio's `STOREYPATH_VISION_KEY`). Without one the helper does not start, unless `STOREYPATH_HELPER_OPEN=1` (a network nothing else can reach) |
| `STOREYPATH_HELPER_SLOTS` | questions answered at once (vLLM 16, llama.cpp 2; with llama.cpp each one more needs more GPU memory). Studio's `STOREYPATH_VISION_PARALLEL` should match it |
| `STOREYPATH_HELPER_GPU_MEMORY` | vLLM: the GPU memory it is given, in GB (`44`) or as a share of the card (`0.5`); default 44 GB, and it uses about 2 GB more. See the table above |
| `STOREYPATH_HELPER_CONTEXT` | tokens of context per question (default 8192) |
| `STOREYPATH_HELPER_CERT`, `STOREYPATH_HELPER_CERT_KEY` | HTTPS with this certificate and key (PEM, mounted into the container, readable by its user, uid 10001); without them plain HTTP, for a trusted network or behind a proxy that speaks HTTPS. Studio checks the certificate (`STOREYPATH_VISION_CA` for your organization's, `STOREYPATH_VISION_INSECURE=1` for a self-signed one on a trusted network) |
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
(BSD-3-Clause), the other Python packages of vLLM's image (each listed with its licence),
NVIDIA's CUDA libraries as vLLM's image ships them (NVIDIA Deep Learning Container
License) and the model's weights (Gemma 4 or Qwen 3.5: Apache-2.0). Each image keeps its
licences in `/usr/share/storeypath/licenses`.
