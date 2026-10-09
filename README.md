# StoreyPath GPU helper

The vision model that [StoreyPath Studio](https://github.com/StoreyPath/storeypath-studio)
asks about the plans, served on a machine with an NVIDIA GPU. Studio reads a drawing
by rules and a small language model on its own; with a helper it also shows each
room's picture to a vision model (Gemma 4 31B) for the judgement calls: what a room is,
whether a door is a door. The helper is optional, and separate from Studio: one Studio
can use several helpers, and several Studios one.

It is llama.cpp's `llama-server`, built for CUDA, with the model baked in: an
OpenAI-compatible API on port 8105, every call but `/health` needing a key. Nothing is
downloaded when it runs, so it works on a machine with no network.

## Build (on a connected machine)

```sh
./fetch-vision.sh                                         # once: about 19 GB, checksum-verified, into models/vision/
docker build --platform linux/amd64 -t storeypath/gpu-helper .
docker save storeypath/gpu-helper | gzip -1 > storeypath-gpu-helper.tar.gz
```

The image is large (the model is inside it), so it is built here, not by CI.

## Run (on the GPU machine)

Linux with Docker, the NVIDIA Container Toolkit and an NVIDIA driver 525 or newer; one
GPU with 32 GB free.

```sh
docker load -i storeypath-gpu-helper.tar.gz
docker run -d --name storeypath-gpu --gpus all -p 8105:8105 \
    -e STOREYPATH_HELPER_KEY=<a long random key, e.g. from openssl rand -hex 32> storeypath/gpu-helper
```

It takes a minute or two to load the model: `/health` answers 503 until then, 200
after. Choose the card with `--gpus '"device=1"'`.

Then tell Studio where it is: on Studio's *GPU helpers* page (admins), or when starting
Studio with `-e STOREYPATH_VISION_URL=http://<gpu machine>:8105/v1 -e
STOREYPATH_VISION_KEY=<the same key>` (several helpers: their addresses separated by
commas; Studio spreads a floor's rooms over them). Studio's `docker/compose.yml` runs
Studio and a helper on one machine.

## Settings

| Variable | |
|---|---|
| `STOREYPATH_HELPER_KEY` | the key every call must carry (Studio's `STOREYPATH_VISION_KEY`). Without one the helper does not start, unless `STOREYPATH_HELPER_OPEN=1` (a network nothing else can reach) |
| `STOREYPATH_HELPER_SLOTS` | questions answered at once (default 2; each one more needs more GPU memory). Studio's `STOREYPATH_VISION_PARALLEL` should match it |
| `STOREYPATH_HELPER_CONTEXT` | tokens of context per question (default 8192) |
| `STOREYPATH_HELPER_CERT`, `STOREYPATH_HELPER_CERT_KEY` | HTTPS with this certificate and key (PEM, mounted into the container); without them plain HTTP, for a trusted network or behind a proxy that speaks HTTPS. Studio checks the certificate (`STOREYPATH_VISION_CA` for your organization's, `STOREYPATH_VISION_INSECURE=1` for a self-signed one on a trusted network) |
| `STOREYPATH_HELPER_NAME` | the name the model is served under (default `gemma-4-31B-it-Q4_K_M`). Studio files answers by it: helpers serving the same model use the same name |
| `STOREYPATH_HELPER_ENGINE` | `llama.cpp` (this image). vLLM may replace it if measurements choose it: see the [Dockerfile](Dockerfile) |

Details are in [start.sh](start.sh) and the [Dockerfile](Dockerfile).

## Licence

Apache 2.0 ([LICENSE](LICENSE)). Inside the image: llama.cpp (MIT), OpenSSL
(Apache-2.0), NVIDIA's CUDA runtime and cuBLAS (CUDA Toolkit EULA, redistributable) and
Gemma 4's weights (Apache-2.0); their licences are in `/usr/share/storeypath/licenses`.
