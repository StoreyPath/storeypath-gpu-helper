#!/bin/sh
# Download the vision model baked into the GPU helper's image, from Hugging Face at a
# pinned revision, each file checked against its checksum below:
#
#   ./fetch-vision.sh [gguf]   for the llama.cpp image (Dockerfile): Gemma 4 31B, 4-bit
#                              (about 18 GB), and its image encoder (about 1 GB), into
#                              models/vision/
#   ./fetch-vision.sh fp8      for the vLLM image (Dockerfile.vllm): Gemma 4 31B, FP8
#                              (about 33 GB), into models/vllm/gemma-4-31B-it-FP8/
#   ./fetch-vision.sh qwen-fp8 for the vLLM image built with MODEL=Qwen3.5-27B-FP8: Qwen
#                              3.5 27B, FP8 (about 29 GB), into models/vllm/Qwen3.5-27B-FP8/
#   ./fetch-vision.sh all      all three
#
# models/ is git-ignored. Building an image needs its model there; running it needs no
# network at all. Downloaded files are kept (a second run fetches only what is missing)
# and checked each time.
set -eu
cd "$(dirname "$0")"
top="$(pwd)"

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

fetch() { # file sha256, from $REPO into the current folder
  if [ -s "$1" ]; then
    echo "have $1"
  else
    echo "fetching $1"
    curl -fL --retry 3 -C - -o "$1.part" "$REPO/$1"
    mv "$1.part" "$1"
  fi
  got="$(sha256 "$1")"
  if [ "$got" != "$2" ]; then
    echo "checksum mismatch for $1: $got (expected $2)" >&2
    exit 1
  fi
  echo "checksum ok: $1"
}

into() { # folder under models/
  mkdir -p "$top/models/$1"
  cd "$top/models/$1"
}

gguf() {
  into vision
  REPO=https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/resolve/main
  fetch gemma-4-31B-it-Q4_K_M.gguf 38bd64c852c4b460434cc7162fa9bdcf242faf86502581a754cb72956bb17f84
  fetch mmproj-F16.gguf 6edcca228213c28d3567a35d22f849eea52d8360875093851959adf5d2f270eb
  ls -lh ./*.gguf
}

# Red Hat's FP8 (dynamic) quantization of Google's Gemma 4 31B (the model card and
# every file vLLM reads; not the repository's benchmark results)
fp8() {
  into vllm/gemma-4-31B-it-FP8
  REPO=https://huggingface.co/RedHatAI/gemma-4-31B-it-FP8-dynamic/resolve/d4ab4f579dd3516f97d8a6a4c98d0653480bad15
  fetch README.md 61792c089e54f8ff500567b0ce1a6bafb499546319ef3c3d176c4db091aee6c9
  fetch chat_template.jinja 6a1015c47ccfcfa67c3b772385bccee357a4d37c3cda37bd202e9047f391ab82
  fetch config.json ffc8a810f1d2e11dfd90756582ca5dee820e4f535ddf4423ec6d2c03256fc321
  fetch generation_config.json d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de
  fetch model-00001-of-00002.safetensors 144030ca3dd3d1cc3960a819788d81ea377d290f4b8d2ed102d94a884d78045e
  fetch model-00002-of-00002.safetensors 1cc6bb1df5374a9899ff7cc0cabd5741b126e5ef892185a8f355f7139a100cf7
  fetch model.safetensors.index.json d4aff3b976d69c123a29d1c085d7ba4de1ac3f4ca1726a7f81e1b11462a64ea2
  fetch processor_config.json 32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c
  fetch tokenizer.json cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f
  fetch tokenizer_config.json 9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633
  du -sh .
}

# the Qwen team's own FP8 of Qwen 3.5 27B
qwen_fp8() {
  into vllm/Qwen3.5-27B-FP8
  REPO=https://huggingface.co/Qwen/Qwen3.5-27B-FP8/resolve/97f5941bf617e31c5e237364a8602ce3f03a551a
  fetch LICENSE 50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32
  fetch README.md c0d0fdb510cbf32ca9043d391fcf33d4c3ff2215e4235128823c9722ab3dece4
  fetch chat_template.jinja a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715
  fetch config.json 53f7e06eebbfe64e70eb0b0979f3da31de9aff6870594e9cf731ba3961501adc
  fetch configuration.json 2d4464e2ead06bc9bc718c781309ad1e7baded626d66e8dcdc8b469ba185faf0
  fetch generation_config.json 303aba891d66ab63908a7b3cc9163bcb835fdf8b9f6301c73216f3f1eb3992dd
  fetch merges.txt a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d
  fetch model.safetensors-00001-of-00011.safetensors 7701198b29eef4e7af4507beeb9d9a005810fc96c1163cc9d978612a7fd08af4
  fetch model.safetensors-00002-of-00011.safetensors 2f43f66745253bb757636ca95d9c0f3e9fc79d23e8426d269e1264304b890583
  fetch model.safetensors-00003-of-00011.safetensors 7aa87a91b2565305dc26f4d839f4449918f9dc63ca4a71819bd652cdd2183b1f
  fetch model.safetensors-00004-of-00011.safetensors b3acefabc2da65dfda0291d815bdf68d7ba104c4d8584ddca220929e93bcbe47
  fetch model.safetensors-00005-of-00011.safetensors 5c3ac92c53fb83e767defdbfa1b42d0085c2f5c5eb514f5f23c4ca550a733d90
  fetch model.safetensors-00006-of-00011.safetensors a8d7c096d2963440a63e33afc6f503e4fd6ada0fff9b42d2dc5ae0bb07df5847
  fetch model.safetensors-00007-of-00011.safetensors 6ab1b1bb6a505b9813a2ad1e5f0efe30aa78aed0abc1dc01564442c64c83501e
  fetch model.safetensors-00008-of-00011.safetensors d76a94ba758a8e26057274a2e6ca28aa9026c0deac34a282bf9bbaf97fbf2e65
  fetch model.safetensors-00009-of-00011.safetensors ea67474a59d4aeed2daaf1b30620e34be78981171f0d058ab6244ee1bd20a3c0
  fetch model.safetensors-00010-of-00011.safetensors 812b25e59727647a4af8f1d5f2c6d95cd7b1c1674f7e8610461ad5e73861779f
  fetch model.safetensors-00011-of-00011.safetensors 7b64e359bd951e90c8dcb609df16e1334cc68a8501f242b7fd483a4398f4b64a
  fetch model.safetensors.index.json 1f48d8a182a2b5dad8e7e3ab4aa661bb5ab4cfc5f54b2c7fdc02d5e00eec9171
  fetch preprocessor_config.json 27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516
  fetch tokenizer.json 5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42
  fetch tokenizer_config.json 316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8
  fetch video_preprocessor_config.json 7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13
  fetch vocab.json ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003
  du -sh .
}

case "${1:-gguf}" in
gguf) gguf ;;
fp8) fp8 ;;
qwen-fp8) qwen_fp8 ;;
all) (gguf) && (fp8) && (qwen_fp8) ;;
*)
  echo "usage: $0 [gguf|fp8|qwen-fp8|all]: the model of the llama.cpp image (gguf, the default), of the vLLM image (fp8), the vLLM image's other choice (qwen-fp8), or all three" >&2
  exit 2
  ;;
esac
