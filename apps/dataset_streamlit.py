import os

import pandas as pd
import streamlit as st

title = 'Training Data Explorer'
st.set_page_config(layout="wide", page_title=title)

st.write(f"""
# {title}
Utility for previewing training dataset
""")

with st.sidebar:
    files = []
    for f in os.listdir('/app/training-dataset'):
        if '.csv' in f:
            files.append(f)

    st.title('Training Dataset')
    training_dataset_file = st.selectbox(
        "Existing Files",
        files,
        index=None,
        placeholder='Select a file to view the content',
    )

    uploaded_files = st.file_uploader("Upload CSV file/s to add to the training dataset.", type="csv",
                                      accept_multiple_files=True)
    # if uploaded_file is not None:
    #     st.write("Uploaded file")

if training_dataset_file is not None:
    st.write(f'Showing preview of ```{training_dataset_file}```')
    df = pd.read_csv(f'/app/training-dataset/{training_dataset_file}')
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode('utf-8'),
        training_dataset_file,
        "text/csv",
        key='download-csv'
    )
    table = st.table(df)

if uploaded_files is not None and len(uploaded_files) != 0:
    for uploaded_file in uploaded_files:
        pd.read_csv(uploaded_file).to_csv(f'/app/training-dataset/{uploaded_file.name}', index=False)
