FROM python:3.10.14-slim-bullseye

WORKDIR /app

RUN apt-get update \
    && apt-get install -y git build-essential zip procps libgl1 jq curl wget poppler-utils libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt
RUN jupyter labextension disable "@jupyterlab/apputils-extension:announcements"

RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
RUN rm -rf /root/.cache/huggingface/hub # delete older models if any

ENV SHELL "/bin/bash"
ENV BASE_MODEL "unsloth/llama-3-8b-Instruct-bnb-4bit"

RUN huggingface-cli download ${BASE_MODEL}

COPY ./config.json /app/config.json
COPY ./llama3_playground /app/llama3_playground
COPY ./training-dataset /app/data/training-dataset
COPY ./models /app/models
COPY ./supervisord.conf /supervisord.conf
COPY build_wheel.py /app/
COPY setup.py /app/
RUN cd /app && python build_wheel.py && pip install dist/llama3_playground-0.0.1-py3-none-any.whl
RUN rm -rf /app/requirements.txt /app/build_wheel.py /app/setup.py /app/dist /app/*.egg-info /app/version.json
RUN mkdir -p /app/logs/
RUN mkdir -p /app/data/

ENV EASY_OCR_MODELS_DIR "/app/models/easyocr"

#RUN mkdir -p $EASY_OCR_MODELS_DIR &&  \
#    cd $EASY_OCR_MODELS_DIR &&  \
#    wget -O craft_mlt_25k.zip https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip &&  \
#    unzip craft_mlt_25k.zip &&  \
#    rm craft_mlt_25k.zip &&  \
#    wget -O english_g2.zip https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip &&  \
#    unzip english_g2.zip &&  \
#    rm english_g2.zip

RUN mkdir -p $EASY_OCR_MODELS_DIR &&  \
    cd $EASY_OCR_MODELS_DIR &&  \
    unzip craft_mlt_25k.zip &&  \
    rm craft_mlt_25k.zip &&  \
    unzip english_g2.zip &&  \
    rm english_g2.zip

RUN wandb disabled

RUN cd /app/ && python -m venv lblstudio && /app/lblstudio/bin/python -m pip install label-studio

RUN echo '#!/bin/bash' >> /start-lbl-studio.sh
RUN echo 'export SECRET_KEY='' && source /app/lblstudio/bin/activate && label-studio start --port 8887' >> /start-lbl-studio.sh

RUN echo '#!/bin/bash' >> /start-services.sh
RUN mkdir -p $(jq -r .models_dir /app/config.json)
RUN echo 'supervisord -c /supervisord.conf' >> /start-services.sh
RUN echo 'wandb disabled' >> /start-services.sh
RUN echo '# cd /app/llama3_playground && uvicorn server.app:app --port 8070 --host 0.0.0.0'
RUN echo 'cd /app/llama3_playground && nohup gunicorn server.app:app --keep-alive 3600 --timeout 3600 --graceful-timeout 300 --threads 10 --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8070 > /app/logs/app-server.log 2>&1 &' >> /start-services.sh
RUN echo 'cd /app' >> /start-services.sh
RUN echo '#nohup streamlit run /app/ocr_streamlit.py --server.port 8885 --server.headless true --browser.gatherUsageStats false > /app/logs/ocr_streamlit.log 2>&1 &' >> /start-services.sh
RUN echo '#nohup streamlit run /app/dataset_streamlit.py --server.port 8886 --server.headless true --browser.gatherUsageStats false > /app/logs/dataset_streamlit.log 2>&1 &' >> /start-services.sh
RUN echo '#nohup streamlit run /app/infer_streamlit.py --server.port 8887 --server.headless true --browser.gatherUsageStats false > /app/logs/infer_streamlit.log 2>&1 &' >> /start-services.sh
RUN echo 'nohup bash /start-lbl-studio.sh > /app/logs/label-studio.log 2>&1 &' >> /start-services.sh
RUN echo "jupyter lab --allow-root --ip=0.0.0.0 --NotebookApp.password='' --NotebookApp.token='' --no-browser" >> /start-services.sh



ENTRYPOINT ["bash", "/start-services.sh"]