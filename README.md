# Llama 3 Playground

A fully-contained environment to finetune Llama 3 model with custom dataset and run inference on the finetuned models

### Requirements

- Docker
- Nvidia GPU

> Note: This is tested only on NVIDIA RTX 2080 and NVIDIA Tesla T4 GPUs so far. It hasn't been tested with the other GPU
> classes or on CPUs.

### Setup

```shell
git checkout https://github.com/amithkoujalgi/llama3-playground.git
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

#### Install NVIDIA Container Toolkit on Ubuntu

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
  "prompt_text": "Respond in a musical and Shakespearean tone"
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
