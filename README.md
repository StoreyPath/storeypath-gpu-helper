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
network. The vLLM image carries the model twice, a full copy and a small one for cards
of 24 or 32 GB, and takes the one the GPU holds. What the key and HTTPS protect, and what
they don't: [Security](#security).

## Which image

| | **vLLM** (recommended): `Dockerfile.vllm` | **llama.cpp**: `Dockerfile` |
|---|---|---|
| Model | Gemma 4 31B twice: in FP8 (8-bit), served as `gemma-4-31B-it-FP8`, and in 4 bits (Intel's AutoRound), served as `gemma-4-31B-it-int4-AutoRound`, for cards of 24 or 32 GB. It takes the FP8 copy where it fits ([Which copy](#which-copy)) | Gemma 4 31B, 4-bit (Q4_K_M), served as `gemma-4-31B-it-Q4_K_M` |
| NVIDIA driver | 580 or newer (CUDA 13.0); a B300 580.82 or newer; on a data-center GPU (A100, H100, H200, B200, L40S…) also the 535 and 570 branches, through the CUDA forward-compatibility libraries inside the image. A GeForce or RTX PRO card with a driver 575 to 579: the CUDA 12.9 build. Each GPU: the table below | 525 or newer |
| GPU | compute capability 8.0 or newer: RTX 4090, RTX 5090, RTX PRO 6000 Blackwell, A100, H100, H200, B200, B300 and others | compute capability 8.0 or newer |
| GPU memory | the FP8 copy: 44 GB by default, about 46 GB used in all; the 4-bit copy: from 21.5 GB, about 1 GB more used in all, so a 24 GB card | about 32 GB with 2 questions at once |
| Questions at once | 16 (`STOREYPATH_HELPER_SLOTS`); with 44 GB about 10 at a time and on a 24 GB card about 5, the rest waiting their turn | 2 |
| Start | two to four minutes (it compiles for the GPU each time) | a minute or two |
| Image | about 53 GB to move (`docker save … \| gzip`), about 76 GB once loaded (its layers; Docker's containerd image store counts about 130, keeping them packed too) | about 21 GB |

### Each GPU

The GPUs it is made for, the image to take, the copy of the model each takes by itself,
and the NVIDIA driver each needs at least. `vllm` is the image built from
`Dockerfile.vllm` as it is (CUDA 13.0); `vllm-cu129` the same built on vLLM's CUDA 12.9
image ([Build](#build-on-a-connected-machine)), for a GeForce or RTX PRO card whose driver
is 575 to 579.

| GPU | Memory | Image | The copy it takes | NVIDIA driver, at least |
|---|---|---|---|---|
| RTX 4090 | 24 GB | `vllm`; `vllm-cu129` with driver 575 to 579 | 4-bit, 4096 tokens a question, about 5 room questions at once | 580 (`vllm-cu129`: 575) |
| RTX 5090 | 32 GB | `vllm`; `vllm-cu129` with driver 575 to 579 | 4-bit, 8192 tokens a question, all 16 at once | 580 (`vllm-cu129`: 575) |
| RTX PRO 6000 Blackwell | 96 GB | `vllm`; `vllm-cu129` with driver 575 to 579 | FP8 | 580 (`vllm-cu129`: 575) |
| H100 | 80 or 94 GB | `vllm` | FP8 | 580, or 535 or 570 |
| H200 | 141 GB | `vllm` | FP8 | 580, or 535.129 or 570 |
| B300 | about 288 GB | `vllm` (`vllm-cu129` has no FP8 kernels for it) | FP8 | 580.82 |
| A100, B200, L40S, RTX 6000 Ada and others of compute capability 8.0 or newer | | `vllm` | FP8 where 43 GB are free, else 4-bit | 580, or on a data-center GPU 535 or 570 (a B200 570) |

Tried on an RTX PRO 6000 Blackwell (compute capability 12.0, driver 580, CUDA 13.0): both
images, `vllm` and `vllm-cu129`, each copy, with no network (`docker run --network none`),
through HTTPS with the key, and Studio's room question on real plans; and the copy each
card takes by itself, with the card's free memory brought down to an RTX 4090's and an RTX
5090's (about 23.5 and 31.4 GB) by another program holding the rest. And on an RTX 5090
itself (32 GB, driver 595, Ubuntu 26.04), offline: it took the small copy by itself (30.9
GB free: 16 questions at once, 8192 tokens each, Marlin's 4-bit kernels), was ready in
under two minutes and answered in 1.5 s; asked for the full copy, it said it needs about
43 GB free and did not start. And on an H100 (80 GB, driver 580, sharing its GPU with
others), offline: the full copy, given 44 GB, on CUTLASS's FP8 kernels and FlashAttention
4, answered in 0.7 s once ready (about four minutes the first time, FlashAttention's
kernels compiled once); the small copy, chosen by hand, on Machete's 4-bit kernels and its
8-bit cache, also in 0.7 s. An RTX 4090 (8.9) and a B300 (10.3) were not to hand: for
them the table below is read from the image and from vLLM's own choices. The full copy has served Studio on an
H100 before, on the CUDA 12.9 build of an earlier version of this image.

### The kernels each GPU runs

The vLLM image is built on vLLM 0.31.0 for CUDA 13.0, which has kernels for each of
these GPUs. What each one takes, read from the GPU code inside the image (`cuobjdump` on
vLLM's kernels and FlashInfer's) and from vLLM's own choices for Gemma 4 on it:

| GPU | Compute capability | FP8 (the full copy) | 4-bit (the small copy) | Attention |
|---|---|---|---|---|
| A100 | 8.0 | weight-only: the 8-bit weights are widened as each layer reads them (Humming's kernels, compiled for the GPU as the helper starts; Marlin's next in line), the arithmetic in BF16 | Marlin | Triton |
| RTX 4090, L40S, RTX 6000 Ada | 8.9 | native FP8 (CUTLASS, Ada's kernels) | Marlin | Triton |
| H100, H200 | 9.0 | native FP8 (CUTLASS, Hopper's kernels) | Machete (Hopper's kernels for 4-bit weights) | Triton for the steps that read a picture, FlashAttention 4 for the rest (with the small copy's 8-bit KV cache, FlashAttention 3 or 4 by layer) |
| B200 | 10.0 | native FP8 (CUTLASS, Blackwell's family kernels, sm_100f) | Marlin, compiled by the driver from its PTX at the first start | Triton for the steps that read a picture, FlashInfer (TensorRT-LLM's kernels) for the rest |
| B300 | 10.3 | as a B200: the same family kernels run on it | as a B200 | as a B200 (FlashInfer has kernels of its own for it) |
| RTX 5090, RTX PRO 6000 Blackwell | 12.0 | native FP8 (CUTLASS, sm_120f) | Marlin | Triton |

The vision tower (the part of Gemma that looks at the picture) runs on PyTorch's own
attention on every GPU. The first start on a GPU also compiles kernels for it (Triton's,
FlashAttention 4's, Humming's) and so takes longer. The CUDA 12.9 build has the same
kernels for every GPU above but the B300: its Blackwell FP8 kernels are for the B200's
chip alone (sm_100a), which a B300 cannot run. [smoke-test.sh](#smoke-test) tries an image
on a GPU.

### Which copy

By itself (`STOREYPATH_HELPER_MODEL=auto`, the default) the helper looks at the memory
it may use: `STOREYPATH_HELPER_GPU_MEMORY` when that gives GB or a share of the card,
else what is free on the GPU as it starts, less what vLLM takes beyond what it is given
(2 GB for the full copy, 1 for the small). With 41 GB or more it serves the full copy,
with less the small one, which needs 21.5 GB; with less than that it does not start, and
says how much is free. `STOREYPATH_HELPER_MODEL=full` or `small` takes that copy whatever
is free. On an empty card:

| | Free | The copy | Given | Tokens a question | Room questions at once |
|---|---|---|---|---|---|
| RTX 4090 | about 23.5 GB | 4-bit | 22.5 GB | 4096 | about 5 |
| RTX 5090 | about 31.4 GB | 4-bit | 30.4 GB | 8192 | 16 |
| 46 GB or more | | FP8 | 44 GB | 8192 | about 10 |

The two copies are served under their own names, as Studio files each answer by the
model's name: a helper that changes copy has every room asked again. The small copy keeps
its memory for the questions (vLLM's KV cache) in 8 bits rather than 16, which fits twice
as many questions at once and changed no more answers than running it again does, and
reads at most 2048 tokens in one step (a room's picture and question are about 1,450),
which leaves it more memory to start with.

### How well each reads

Both engines read rooms as well as each other; vLLM reads them two to four times faster.
Take the llama.cpp image where the driver is older than vLLM's CUDA 12.9 build needs.

Measured on one NVIDIA RTX PRO 6000 Blackwell (96 GB, the 300 W Max-Q card), on the same
rooms, with Studio's own question (its picture of a room, the choices, a JSON answer):

| Engine and model | Right | Rooms a minute, one at a time | Rooms a minute, 16 at once |
|---|---|---|---|
| vLLM, Gemma 4 31B FP8 | 81% | 53–55 | 87–91 |
| vLLM, Qwen 3.5 27B FP8 (`MODEL=Qwen3.5-27B-FP8`, below) | 82% | 83 | 316–323 |
| llama.cpp, Gemma 4 31B 4-bit | 82% | about 22 | about 22 (whatever the parallelism) |

The vLLM image's small copy against its full one, on the same card given the memory of a
smaller one, on over a thousand rooms of real plans (Studio's pictures and its question).
The full copy asked twice does not answer every room the same (vLLM batches the questions
differently each time), which is the yardstick:

| | The small copy as an RTX 4090 takes it (22.5 GB) | As an RTX 5090 takes it (30.4 GB) | The full copy, asked again |
|---|---|---|---|
| The same room type as the full copy | 82% | 82% | 91% |
| The same answer about the red outline (one room, merged, part of one, none) | 94% | 94% | 97% |
| Right, of the rooms whose drawing writes what they are (the full copy: 69%) | 68% | 67% | 69% |
| Rooms a minute, 16 sent at once (on this card) | 72 | 72 | 87 |

They differ most between an office and another room, and on what is no room at all; on
bathrooms, stairs, corridors, lifts, kitchens and lobbies the two copies agree nine times
in ten or more.

Why Intel's copy: it is the smallest 4-bit Gemma 4 31B published, its weights in groups of
128 (19.2 GB). Those in groups of 32, Google's own (trained for 4 bits) and Red Hat's
NVFP4, are 20.4 GB, which by our sums leaves a 24 GB card too little memory for one
question of 4096 tokens. Of the two in groups of 128, Intel's agreed with the full copy on
83% of rooms, the other on 70%.

## Build (on a connected machine)

```sh
# vLLM
./fetch-vision.sh vllm                                    # once: about 51 GB, checksum-verified, into models/vllm/
docker build --platform linux/amd64 -f Dockerfile.vllm -t storeypath/gpu-helper:vllm .
docker save storeypath/gpu-helper:vllm | gzip -1 > storeypath-gpu-helper-vllm.tar.gz

# llama.cpp
./fetch-vision.sh                                         # once: about 19 GB, checksum-verified, into models/vision/
docker build --platform linux/amd64 -t storeypath/gpu-helper .
docker save storeypath/gpu-helper | gzip -1 > storeypath-gpu-helper.tar.gz
```

`./fetch-vision.sh vllm` fetches both copies of the model: `fp8`, the full one (about
33 GB), and `small`, the 4-bit one (about 19 GB); each can be fetched alone.

The images are large (the models are inside), so they are built here, not by CI. Each
Dockerfile has its own build context (`.dockerignore`, `Dockerfile.vllm.dockerignore`,
which BuildKit, Docker's builder by default, reads): only its models are sent to Docker.
Building the vLLM image takes half an hour or more, most of it compressing the models'
layers.

For a GeForce or RTX PRO card whose NVIDIA driver is 575 to 579 (they have no forward
compatibility, and the image's CUDA 13.0 needs 580), build on vLLM's CUDA 12.9 image
instead. It serves every GPU in the tables above but a B300, whose FP8 kernels it lacks:

```sh
docker build --platform linux/amd64 -f Dockerfile.vllm -t storeypath/gpu-helper:vllm-cu129 \
    --build-arg VLLM_IMAGE=vllm/vllm-openai:v0.31.0-cu129@sha256:ff29e51a9457f191deb332b96fa6440584fa941aa2a548449b604e2b1e5e1eeb .
```

The vLLM image can carry Qwen 3.5 27B (FP8) as its full copy instead of Gemma's:
`./fetch-vision.sh qwen-fp8`, then build with `--build-arg MODEL=Qwen3.5-27B-FP8` (and
e.g. `-t storeypath/gpu-helper:vllm-qwen`). It is served as `Qwen3.5-27B-FP8`; Studio
files answers by the model's name, so it asks every room again. The small copy stays
Gemma's.

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

The vLLM helper takes the copy of the model the card holds ([Which copy](#which-copy));
its log's first line says which, and why. To have one regardless, add `-e
STOREYPATH_HELPER_MODEL=full` or `small`.

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
written in it. It prints the GPU, the copy of the model the helper took and what vLLM
chose for it (attention, FP8 or 4-bit kernels), the answer and how long each step took,
and removes its container.

```sh
./smoke-test.sh storeypath/gpu-helper:vllm 0                     # the GPU nvidia-smi numbers 0
./smoke-test.sh storeypath/gpu-helper:vllm nvidia.com/gpu=0      # with the toolkit's CDI
STOREYPATH_HELPER_GPU_MEMORY=auto ./smoke-test.sh storeypath/gpu-helper:vllm 1   # a GPU others use too
STOREYPATH_HELPER_GPU_MEMORY=22 ./smoke-test.sh storeypath/gpu-helper:vllm 0     # the small copy, as a 24 GB card takes it
./smoke-test.sh storeypath/gpu-helper:vllm 0 --linear-backend marlin   # arguments after the GPU go to vLLM
```

It makes a key of its own for the test unless `STOREYPATH_HELPER_KEY` is set, and never
prints it. `STOREYPATH_HELPER_MODEL`, `_GPU_MEMORY`, `_CONTEXT` and `_SLOTS` are passed
on when set, and `SMOKE_DOCKER_ARGS` adds options to `docker run` (e.g. `--cpuset-cpus
0-31 --memory 64g` on a machine busy with other work). The first start on a GPU compiles
kernels for it; it waits up to half an hour (`SMOKE_TIMEOUT`, in seconds).

On a GPU other programs share, the vLLM helper takes the memory it is given
(`STOREYPATH_HELPER_GPU_MEMORY`) and about 2 GB more (the small copy about 1); that much
must be free when it starts. Given nothing, it takes the copy and the memory that what is
free allows, and stops with a message saying how much is free when that is too little for
either. Measured with Studio's question, the full copy:

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

The small copy, on the same card:

| `STOREYPATH_HELPER_GPU_MEMORY` | GPU memory used (nvidia-smi) | Tokens a question | Questions at once | Rooms a minute, 16 sent at once |
|---|---|---|---|---|
| 22 | 22.7 GB | 4096 | 3–4 | 70 |
| 22.5 (an RTX 4090's free memory, less 1 GB) | 23.2 GB | 4096 | 5–6 | 72 |
| 30.4 (an RTX 5090's free memory, less 1 GB) | under the 31.4 GB free | 8192 | 16 | 72 |

## Moving it to an offline machine

The image holds all it needs: the model, vLLM (or llama.cpp) and the helper. Nothing is
downloaded when it starts or runs. Prepare it on the machine with internet:

1. Build it ([Build](#build-on-a-connected-machine)).
2. Try it with no network, on a GPU of that machine: `./smoke-test.sh storeypath/gpu-helper:vllm 0`
   ([Smoke test](#smoke-test)). When it answers, nothing it needs is outside the image.
3. Save it, and note the file's checksum to check after the move:

   ```sh
   docker save storeypath/gpu-helper:vllm | gzip -1 > storeypath-gpu-helper-vllm.tar.gz
   shasum -a 256 storeypath-gpu-helper-vllm.tar.gz > storeypath-gpu-helper-vllm.tar.gz.sha256
   ```

On the offline machine, which needs Linux, Docker, the NVIDIA Container Toolkit and the
driver the image asks for ([Which image](#which-image)):

1. Check the file and load it:

   ```sh
   shasum -a 256 -c storeypath-gpu-helper-vllm.tar.gz.sha256
   docker load -i storeypath-gpu-helper-vllm.tar.gz
   ```

2. Make its key and start it ([Run](#run-on-the-gpu-machine)). Its first start compiles
   kernels for that GPU, with no network, for 10 to 30 minutes; `/health` answers 200 once
   it is ready. They are kept with the container: one made again compiles them again.

What the file does not hold: the key (`helper.env`, made on the offline machine, given
as it starts), its certificate (made as it starts) and anything of Studio's (the helper
keeps nothing it is asked).

## Studio

Tell Studio where the helper is: on Studio's *GPU helpers* page (admins), or when
starting Studio:

| Studio's variable | |
|---|---|
| `STOREYPATH_VISION_URL` | `https://<gpu machine>:8105/v1` (`http://` with `STOREYPATH_HELPER_PLAIN_HTTP=1`); several helpers: their addresses separated by commas, and Studio spreads a floor's rooms over them |
| `STOREYPATH_VISION_KEY` | the helper's `STOREYPATH_HELPER_KEY` |
| `STOREYPATH_VISION_PARALLEL` | the helper's slots: 16 for vLLM, 2 for llama.cpp |
| `STOREYPATH_VISION_MODEL` | the served name: `gemma-4-31B-it-FP8` (vLLM, the full copy), `gemma-4-31B-it-int4-AutoRound` (vLLM, the small copy), `gemma-4-31B-it-Q4_K_M` (llama.cpp). Studio files answers by it: rooms read by one model are asked again of another |
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
| `STOREYPATH_HELPER_MODEL` | vLLM: `auto` (the default: the full copy where the memory it may use is 41 GB or more, else the small one, [Which copy](#which-copy)), `full`, `small`, or a model folder of your own in Hugging Face's format. llama.cpp: the GGUF file (the image sets it) |
| `STOREYPATH_HELPER_GPU_MEMORY` | vLLM: the GPU memory it is given, in GB (`44`), as a share of the card (`0.5`), or `auto` (what is free when it starts, less 2 GB, up to what 16 questions at once need: 52 for the full copy, 31 for the small; for a GPU others use too). Not given: the full copy 44 GB, the small one what is free less 1 GB up to 31, and less where less is free; it does not start with less than the full copy's 41 or the small one's 21.5. It uses about 2 GB more than it is given (the small copy about 1). See the tables above |
| `STOREYPATH_HELPER_CONTEXT` | tokens of context per question (default 8192; the small copy given less than 24 GB: 4096) |
| `STOREYPATH_HELPER_CERT`, `STOREYPATH_HELPER_CERT_KEY` | optional: HTTPS with this certificate and key (PEM, mounted into the container, readable by its user, uid 10001), e.g. from your organisation's CA, which Studio then checks (`STOREYPATH_VISION_CA`). Without them the helper makes its own as it starts (self-signed, EC P-256, ten years; its SHA-256 fingerprint in the log), which encrypts but which Studio cannot check ([Security](#security)) |
| `STOREYPATH_HELPER_PLAIN_HTTP` | `1`: plain HTTP, no certificate: for an SSH tunnel, or a proxy in front that speaks HTTPS ([Security](#security)) |
| `STOREYPATH_HELPER_NAME` | the name the model is served under (default: its folder's or file's, `gemma-4-31B-it-FP8` or `gemma-4-31B-it-int4-AutoRound` for vLLM, `gemma-4-31B-it-Q4_K_M` for llama.cpp). Studio files answers by it: helpers serving the same model use the same name |
| `STOREYPATH_HELPER_IMAGE_KWARGS` | vLLM: how the model's image processor reads a picture, as JSON; for Gemma `{"max_soft_tokens":1120}` unless given (up to 1,120 tokens a picture, as measured; Gemma's own default, 280, is coarse for a plan), `{}` for the model's own |
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
Learning Container License) and the models' weights: Gemma 4 (Apache-2.0, Google) in
Red Hat's FP8 quantization (Apache-2.0) and Intel's 4-bit one (AutoRound, under the
original model's licence, so Apache-2.0), or Qwen 3.5 (Apache-2.0). Each image keeps its
licences in `/usr/share/storeypath/licenses`.
