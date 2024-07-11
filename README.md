# Llama 3 Playground

A fully-contained environment to finetune Llama 3 model with custom dataset and run inference on the finetuned models

### Requirements

- Docker
- Nvidia GPU

> Note: This is tested only on NVIDIA RTX 2080 and NVIDIA Tesla T4 GPUs so far. It hasn't been tested with the other GPU
> classes or on CPUs.


Run this command on your host machine to check which Nvidia GPU you've installed.

```shell
nvidia-smi
```

That should display your GPU info.

```shell
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.171.04             Driver Version: 535.171.04   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 2080        Off | 00000000:01:00.0  On |                  N/A |
| 22%   38C    P8              17W / 215W |    197MiB /  8192MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```

### Setup

```shell
git clone https://github.com/amithkoujalgi/llama3-playground.git
cd llama3-playground

bash build.sh
```

### Run

```shell
bash run.sh
```

This starts the Docker container with the following services.

| Service           | Host                       | Description                                                                                     |   |
|-------------------|----------------------------|-------------------------------------------------------------------------------------------------|---|
| Supervisor        | http://localhost:8884      | For running training on custom dataset and viewing logs of trainer process                      |   |
| FastAPI Server    | http://localhost:8883/docs | For accessing APIs of the model server                                                          |   |
| JupyterLab Server | http://localhost:8888/lab  | Access JupyterLab interface for browsing the container and updating/experimenting with the code |   |

### Additional setup instructions

This would be needed if you do not have NVIDIA Container Toolkit installed on your host machine.

#### Install NVIDIA Container Toolkit if you're running a Ubuntu host

```shell
# Configure the production repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Optionally, configure the repository to use experimental packages
sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update the packages list from the repository
sudo apt-get update

# Install the NVIDIA Container Toolkit packages
sudo apt-get install -y nvidia-container-toolkit
```

For other environments, please refer
to [this](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## APIs

### Inference

#### Generate response from the model

```shell
curl --silent -X 'POST' \
  'http://localhost:8883/api/infer/sync/ctx-text' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model_name": "llama-3-8b-instruct-custom-1720690384",
  "context_data": "You are a magician who goes by the name Magica",
  "question_text": "Who are you?",
  "prompt_text": "Respond in a musical and Shakespearean tone",
  "max_new_tokens": 50
}' | jq -r ".data.response"
```

### OCR

#### Get status of OCR process. Returns `true` if any OCR process is running, `false` otherwise.

```shell
curl -X 'GET' \
  'http://localhost:8883/api/ocr/status' \
  -H 'accept: application/json'
```

#### Run OCR on PDF file by uploading the file

```shell
curl -X 'POST' \
  'http://localhost:8883/api/ocr/sync/pdf' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@your_file.pdf;type=application/pdf'
```

References:

- https://huggingface.co/unsloth/llama-3-8b-bnb-4bit
- https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit
- https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
