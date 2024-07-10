import os
import shutil
import subprocess

import streamlit as st

title = 'OCR for PDFs'
st.set_page_config(layout="wide", page_title=title)

st.write(f"""
# {title}
Utility for running OCR on PDF files.
""")

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
        return_code =p.returncode

        if return_code == 0:
            with open('/app/ocr/pdf-text-data.txt', 'r') as f:
                text_data = f.read()
            with open('/app/ocr/ocr-data.json', 'r') as f:
                ocr_data = f.read()
            st.write(f'Raw text from OCR:')
            st.write(f'```{text_data}')
            st.write(f'OCR JSON Data:')
            st.write(f'```json{ocr_data}')
        else:
            st.write(f'Error from OCR: {err}')

    #
    # st.write(f'OCR data with coordinates:')
    # st.write(f'```json{ocr_data}')
