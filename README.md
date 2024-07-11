# Llama 3 Playground

A fully-contained environment to finetune Llama 3 model with custom dataset and run inference on the finetuned models


> Note: This is tested only on NVIDIA RTX 2080 and NVIDIA Tesla T4 GPUs so far. It hasn't been tested with the other GPU
> classes or on CPUs.

### Install NVIDIA Container Toolkit

#### On Ubuntu

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

#### For other environments, please refer to [this](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## APIs

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

- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
