import os
import shutil
import subprocess
from typing import Generator

import streamlit as st

title = 'PDF Chat'
st.set_page_config(layout="wide", page_title=title)
st.title(title)


def ask_model(messages: list) -> Generator[str, None, None]:
    question = messages[-1]['content']
    cmd_arr = ['python', '/app/infer.py', '/app/ocr/pdf-text-data.txt', question]
    out = ""
    err = ""
    p = subprocess.Popen(cmd_arr, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in p.stdout:
        out = out + "\n" + line.decode("utf-8")
    for line in p.stderr:
        err = err + "\n" + line.decode("utf-8")
    p.wait()
    return_code = p.returncode

    if return_code == 0:
        with open('/app/ocr/llm-response.txt', 'r') as f:
            for line in f:
                yield line.strip()
    else:
        yield "Error generating response."


with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload a PDF file.",
        type="pdf",
        accept_multiple_files=False
    )
if uploaded_file is not None:
    runs_dir = "/app/ocr"

    try:
        shutil.rmtree(runs_dir)
    except Exception as e:
        pass

    os.makedirs(runs_dir, exist_ok=True)

    pdf_upload_file = f'{runs_dir}/{uploaded_file.name}'

    bytes_data = uploaded_file.getvalue()
    with open(pdf_upload_file, 'wb') as f:
        f.write(bytes_data)

    st.write(f'Uploaded file: ```{uploaded_file.name}```')

    with st.spinner('Processing...'):
        cmd_arr = ['python', '/app/ocr.py', '/app/ocr', f'{pdf_upload_file}']
        out = ""
        err = ""
        p = subprocess.Popen(cmd_arr, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for line in p.stdout:
            out = out + "\n" + line.decode("utf-8")
        for line in p.stderr:
            err = err + "\n" + line.decode("utf-8")
        p.wait()
        return_code = p.returncode

        if return_code == 0:
            st.write(f'Processed successfully! Ready for your questions.')

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            if prompt := st.chat_input("How can I help you?"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    response = st.write_stream(
                        ask_model(
                            st.session_state.messages
                        )
                    )
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
        else:
            st.write(f'Error in OCR: {err}')
