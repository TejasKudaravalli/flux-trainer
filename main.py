import time

import streamlit as st

from src import (
    create_destination_model,
    create_zip_file_link,
    resize_zip_file,
    start_replicate_training,
    check_training_status,
    get_model_url
)


def extract_flux_name(filename: str) -> str:
    filename = filename.replace(".zip", "").split("_")
    filename = filename[-1]
    return filename.lower()


st.set_page_config(page_title="Flux LoRA Trainer", page_icon="🚀")
st.title("Flux LoRA Trainer")
st.write("Upload a ZIP file containing images to train a Flux LoRA model.")
st.write("Expected File Name: `******_firstnamelastname.zip`")
uploaded_file = st.file_uploader("Drag and drop a .zip file", type=["zip"])

if uploaded_file:
    filename = uploaded_file.name
    flux_name = extract_flux_name(filename)
    result_zip_buffer = resize_zip_file(uploaded_file)
    with st.spinner("Uploading file to Replicate Files API..."):
        zip_url = create_zip_file_link(result_zip_buffer)
        st.success("Training Data added to Replicate for training")

    with st.spinner("Initializing training..."):
        create_destination_model(flux_name=flux_name)
        training_id = start_replicate_training(zip_url, flux_name)

    if training_id:
        st.success(f"Training started! Training ID: `{training_id}`")
        with st.spinner("Training in progress... This may take a while."):
            time.sleep(5)
            training_status = check_training_status(training_id)

        if training_status.get("status") == "succeeded":
            model_url = get_model_url(flux_name)
            st.success(f"🎉 Training complete! Model URL: {model_url}")
        else:
            st.error(f"Training failed. - {training_status.get('error')}")
            st.error(f"{training_status.get('logs')}")
            st.stop()
