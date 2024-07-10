import os
import subprocess
from pathlib import Path
from typing import Optional, Generator

import streamlit as st

title = "PDF Bot"
init_msg = "Hello, I'm your PDF assistant. Upload a PDF to get going."

pdfs_directory = os.path.join(str(Path.home()), 'langchain-store', 'uploads', 'pdfs')
os.makedirs(pdfs_directory, exist_ok=True)

print(f"Using PDFs upload directory: {pdfs_directory}")

st.set_page_config(page_title=title)


def ask_model(question: str) -> Generator[str, None, None]:
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


def on_upload_change():
    # clear_chat_history()
    print("File changed.")
    st.session_state.messages = [{"role": "assistant", "content": init_msg}]


def set_uploaded_file(_uploaded_file: str):
    st.session_state['uploaded_file'] = _uploaded_file


def get_uploaded_file() -> Optional[str]:
    if 'uploaded_file' in st.session_state:
        return st.session_state['uploaded_file']
    return None


with st.sidebar:
    st.title(title)
    st.write('This chatbot accepts a PDF file and lets you ask questions on it.')
    uploaded_file = st.file_uploader(
        label='Upload a PDF', type=['pdf', 'PDF'],
        accept_multiple_files=False,
        key='file-uploader',
        help=None,
        on_change=on_upload_change,
        args=None,
        kwargs=None,
        disabled=False,
        label_visibility="visible"
    )

    if uploaded_file is not None:
        runs_dir = "/app/ocr"
        pdf_upload_file = f'{runs_dir}/{uploaded_file.name}'
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
                added = False
                my_msg = f"Great! Now, what do you want from `{uploaded_file.name}`?"
                for msg in st.session_state.messages:
                    if msg["content"] == my_msg:
                        added = True
                if not added:
                    st.session_state.messages.append({"role": "assistant", "content": my_msg})
                bytes_data = uploaded_file.getvalue()
                target_file = os.path.join(pdfs_directory, uploaded_file.name)
                set_uploaded_file(target_file)
                with open(target_file, 'wb') as f:
                    f.write(bytes_data)
            else:
                st.session_state.messages.append({"role": "assistant", "content": err})


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": init_msg}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    from streamlit_js_eval import streamlit_js_eval
    streamlit_js_eval(js_expressions="parent.window.location.reload()")
    st.session_state.messages = [{"role": "assistant", "content": init_msg}]


st.sidebar.button('Reset', on_click=clear_chat_history)

# User-provided prompt
if prompt := st.chat_input(disabled=False, placeholder="What do you want to know from the uploaded PDF?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    source_file = get_uploaded_file()
    if source_file is None:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                placeholder = st.empty()
                full_response = 'PDF file needs to be uploaded before you can ask questions on it 😟. Please upload a file.'
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                question = dict(st.session_state.messages[-1]).get('content')
                response = ask_model(
                    question=question
                )
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
