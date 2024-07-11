FROM python:3.10.14-slim-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y git build-essential zip procps libgl1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt
RUN jupyter labextension disable "@jupyterlab/apputils-extension:announcements"

RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
RUN rm -rf /root/.cache/huggingface/hub # delete older models if any

ENV SHELL "/bin/bash"
ENV BASE_MODEL "unsloth/llama-3-8b-Instruct-bnb-4bit"

RUN huggingface-cli download ${BASE_MODEL}

RUN rm /app/requirements.txt
RUN mkdir -p /app/logs/
RUN mkdir -p /app/data/
COPY ./config.json /app/config.json
COPY ./core /app/core
COPY ./training-dataset /app/data/training-dataset
COPY ./supervisord.conf /supervisord.conf

RUN apt-get update && apt-get install -y jq curl wget && rm -rf /var/lib/apt/lists/*

# Download EasyOCR models
RUN mkdir -p /app/data/easyocr-models && \
    cd /app/data/easyocr-models &&  \
    wget https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip && \
    unzip craft_mlt_25k.zip && \
    rm craft_mlt_25k.zip && \
    wget https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip && \
    unzip english_g2.zip && \
    rm english_g2.zip

RUN echo '#!/bin/bash' >> /start-services.sh
RUN echo 'supervisord -c /supervisord.conf' >> /start-services.sh
RUN echo 'cd /app/core && nohup gunicorn server.app:app --keep-alive 3600 --timeout 3600 --graceful-timeout 300 --threads 10 --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8070 > /app/logs/app-server.log 2>&1 &' >> /start-services.sh
RUN echo 'cd /app' >> /start-services.sh
RUN echo '#nohup streamlit run /app/ocr_streamlit.py --server.port 8885 --server.headless true --browser.gatherUsageStats false > /app/logs/ocr_streamlit.log 2>&1 &' >> /start-services.sh
RUN echo '#nohup streamlit run /app/dataset_streamlit.py --server.port 8886 --server.headless true --browser.gatherUsageStats false > /app/logs/dataset_streamlit.log 2>&1 &' >> /start-services.sh
RUN echo '#nohup streamlit run /app/infer_streamlit.py --server.port 8887 --server.headless true --browser.gatherUsageStats false > /app/logs/infer_streamlit.log 2>&1 &' >> /start-services.sh
RUN echo "jupyter lab --allow-root --ip=0.0.0.0 --NotebookApp.password='' --NotebookApp.token='' --no-browser" >> /start-services.sh

ENTRYPOINT ["bash", "/start-services.sh"]