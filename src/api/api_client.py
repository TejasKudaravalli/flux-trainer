from io import BytesIO
import time

import requests
import streamlit as st
from loguru import logger


def create_zip_file_link(file) -> str:
    file_content = BytesIO(file.read())
    file_content.seek(0)
    url = "https://api.replicate.com/v1/files"
    headers = {"Authorization": f"Bearer {st.secrets['REPLICATE_API_TOKEN']}"}
    files = {"content": ("data.zip", file_content, "application/zip")}
    response = requests.post(url, headers=headers, files=files)
    if response.status_code in [200, 201]:
        response = response.json()
        zip_url = response.get("urls").get("get")
        logger.info(f"Zip URL: {zip_url}")
        return zip_url
    else:
        raise Exception(f"Upload failed: {response.text}")


def create_destination_model(flux_name: str):
    url = "https://api.replicate.com/v1/models"
    headers = {
        "Authorization": f"Bearer {st.secrets['REPLICATE_API_TOKEN']}",
        "Content-Type": "application/json",
    }
    data = {
        "owner": st.secrets["USER_NAME"],
        "name": f"{flux_name}_flux_lora",
        "description": f"A model for {flux_name}",
        "visibility": "public",
        "hardware": "gpu-l40s",
    }
    response = requests.post(url, headers=headers, json=data)
    logger.info(f"Destination Model Status: {response.status_code}")


def start_replicate_training(zip_url: str, flux_name: str) -> str:
    # model_name is combinination of model_owner and model_name
    MAX_TRAIN_STEPS = 1000
    model_name = st.secrets["MODEL_NAME"]
    model_version = st.secrets["MODEL_VERSION"]
    url = f"https://api.replicate.com/v1/models/{model_name}/versions/{model_version}/trainings"
    logger.info(f"URL: {url}")
    headers = {
        "Authorization": f"Bearer {st.secrets['REPLICATE_API_TOKEN']}",
        "Content-Type": "application/json",
    }
    data = {
        "destination": f"{st.secrets['USER_NAME']}/{flux_name}_flux_lora",
        "input": {
            "input_images": zip_url,
            "trigger_word": flux_name,
            "steps": MAX_TRAIN_STEPS,
        },
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code in [200, 201]:
        response = response.json()
        training_id = response.get("id")
        return training_id
    else:
        raise Exception(f"Failed to start training: {response.text}")


def check_training_status(training_id:str) -> dict:
    while True:
        url = f"https://api.replicate.com/v1/trainings/{training_id}"
        headers = {"Authorization": f"Bearer {st.secrets['REPLICATE_API_TOKEN']}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 404:
            raise Exception(f"training_id: {training_id} not found")
        response = response.json()
        status = response.get("status")
        message_holder = st.empty()
        message_holder.write(f"Current Status: {status}")
        if status in ["succeeded", "failed"]:
            return response
        time.sleep(30)  # Check every 30 seconds
        message_holder.empty()

def get_model_url(flux_name: str) -> str:
    model_owner = st.secrets["USER_NAME"]
    model_name = f"{flux_name}_flux_lora"
    url = f"https://api.replicate.com/v1/models/{model_owner}/{model_name}"
    logger.info(url)
    headers = {"Authorization": f"Bearer {st.secrets['REPLICATE_API_TOKEN']}"}
    response = requests.get(url, headers=headers)
    logger.info(response.status_code)
    if response.status_code in [200, 201]:
        response = response.json()
        model_url = response.get("url")
        return model_url
    else:
        raise Exception(f"Failed to fetch the link: {response.text}")