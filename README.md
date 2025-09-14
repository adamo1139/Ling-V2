# Ling V2 Polish model pretraining


## Introduction

fork of Ling V2 repo, customized to pre-train a similar model but on Polish Fineweb-2 dataset

Your `pip list` should look something like this

```
Package                  Version
------------------------ -------------
accelerate               1.10.1
aiohappyeyeballs         2.6.1
aiohttp                  3.12.15
aiosignal                1.4.0
aniso8601                10.0.1
annotated-types          0.7.0
apex                     0.1
attrs                    25.3.0
bitsandbytes             0.47.0
blinker                  1.9.0
certifi                  2025.8.3
charset-normalizer       3.4.3
click                    8.2.1
datasets                 3.6.0
deep_ep                  1.2.1+1d3963d
dill                     0.3.8
einops                   0.8.1
filelock                 3.19.1
Flask                    3.1.2
Flask-RESTful            0.3.10
frozenlist               1.7.0
fsspec                   2025.3.0
gitdb                    4.0.12
GitPython                3.1.45
hf_transfer              0.1.9
hf-xet                   1.1.10
huggingface-hub          0.34.4
idna                     3.10
importlib_metadata       8.7.0
itsdangerous             2.2.0
Jinja2                   3.1.6
MarkupSafe               3.0.2
megatron-core            0.13.1
ml_dtypes                0.5.3
mpmath                   1.3.0
multidict                6.6.4
multiprocess             0.70.16
networkx                 3.5
ninja                    1.13.0
numpy                    1.26.4
nvidia-cublas-cu12       12.4.5.8
nvidia-cuda-cupti-cu12   12.4.127
nvidia-cuda-nvrtc-cu12   12.4.127
nvidia-cuda-runtime-cu12 12.4.127
nvidia-cudnn-cu12        9.1.0.70
nvidia-cufft-cu12        11.2.1.3
nvidia-cufile-cu12       1.13.1.3
nvidia-curand-cu12       10.3.5.147
nvidia-cusolver-cu12     11.6.1.9
nvidia-cusparse-cu12     12.3.1.170
nvidia-cusparselt-cu12   0.6.2
nvidia-nccl-cu12         2.21.5
nvidia-nvjitlink-cu12    12.4.127
nvidia-nvtx-cu12         12.4.127
onnx                     1.19.0
onnx-ir                  0.1.9
onnxscript               0.3.1
packaging                25.0
pandas                   2.3.2
pip                      25.2
platformdirs             4.4.0
propcache                0.3.2
protobuf                 6.32.1
psutil                   7.0.0
pyarrow                  21.0.0
pybind11                 3.0.1
pydantic                 2.11.9
pydantic_core            2.33.2
python-dateutil          2.9.0.post0
pytz                     2025.2
PyYAML                   6.0.2
regex                    2025.9.1
requests                 2.32.5
safetensors              0.6.2
sentencepiece            0.2.1
sentry-sdk               2.37.1
setuptools               78.1.1
six                      1.17.0
smmap                    5.0.2
sympy                    1.13.1
tiktoken                 0.11.0
tokenizers               0.22.0
torch                    2.6.0
tqdm                     4.67.1
transformer_engine       2.6.0.post1
transformer_engine_cu12  2.6.0.post1
transformer_engine_torch 2.6.0.post1
transformers             4.56.1
triton                   3.2.0
typing_extensions        4.15.0
typing-inspection        0.4.1
tzdata                   2025.2
urllib3                  2.5.0
wandb                    0.21.4
Werkzeug                 3.1.3
wheel                    0.45.1
xxhash                   3.5.0
yarl                     1.20.1
zipp                     3.23.0
```

how to get there?

install Megatron

```
pip install megatron-core==0.13.0 torch==2.6.0
pip install --no-build-isolation transformer-engine[pytorch]
```

At this stage you should add flash-attention to the mix if your machine can stand it

Install nvshmem

nvshmem install guide - https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/nvshmem-install-proc.html

then add device libs for nvshmem

`sudo apt install libnvshmem3-static-cuda-12 nvshmem-cuda-12`

then we install deep_ep

```
git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP

CXXFLAGS="-I/usr/include/nvshmem_12" \
NVSHMEM_DIR=/usr \
NVSHMEM_LIB=/usr/lib/x86_64-linux-gnu/nvshmem/12 \
python setup.py build_ext \
--include-dirs=/usr/include/nvshmem_12 \
--library-dirs=/usr/lib/x86_64-linux-gnu/nvshmem/12 \
install
````

then install apex

```
git clone https://github.com/NVIDIA/apex.git
cd apex

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

you might need to comment out minor version mismatch with CUDA from being checked. 12.4/12.5/12.8 should be FINE, you should use 12.4 in the system as baseline.

get Megatron

```
cd Ling-V2
# apply megatron patch
bash training/megatron/apply_patch.sh
# apply te patch
bash training/te/apply_te_patch.sh
```

patch megatron processing dataset code to support parquet

```
bash deploy_parquet_support.sh
```

to save trained model to safetensors, runs something like this

```
python tools/convert_dcp_to_safetensors.py --checkpoint-path /home/adamo/projects/pretain_pol/Ling-V2/pretrain_512/iter_0000200 --target-path /home/adamo/projects/pretain_pol/Ling-V2/szczypulka2/ --force-bf16 --override-tokenizer-path /home/adamo/projects/pretain_pol/Ling-V2/resource/tokenizer/config_pretrain
```
