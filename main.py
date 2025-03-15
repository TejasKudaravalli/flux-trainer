import streamlit as st
import requests
import os 
import time
from io import BytesIO
from replicate import Client
import base64

REPLICATE_API_KEY = st.secrets["REPLICATE_API_TOKEN"]
MODEL_NAME = st.secrets["MODEL_NAME"]
MODEL_VERSION = st.secrets["MODEL_VERSION"]
MAX_TRAIN_STEPS = 1000
USER_NAME = "micksil"

replicate = Client(
    api_token=REPLICATE_API_KEY,
    headers={
        "User-Agent": "my-app/1.0"
    }
)


def create_zip_file_link(file) -> str:
    file_content = file.read()
    base64_encoded = base64.b64encode(file_content).decode("utf-8")
    file_input = f"data:application/octet-stream;base64,{base64_encoded}"
    return file_input


def extract_flux_name(filename: str) -> str:
    filename = filename.replace(".zip", "").split("_")
    filename = filename[-1]
    return filename.lower()


def create_destination_model(flux_name:str):
    url = "https://api.replicate.com/v1/models"
    headers = {
        "Authorization": f"Bearer {REPLICATE_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "owner": USER_NAME,
        "name": f"{flux_name}_flux_lora",
        "description": f"A model for flux_name",
        "visibility": "public",
        "hardware": "gpu-l40s"
    }
    response = requests.post(url, headers=headers, json=data)



def start_replicate_training(zip_url:str, flux_name:str) -> str:
    # model_name is combinination of model_owner and model_name
    url = f"https://api.replicate.com/v1/models/{MODEL_NAME}/{MODEL_VERSION}/trainings"
    headers = {
        "Authorization": f"Bearer {REPLICATE_API_KEY}",
        "Content-Type": "application/json"
    }
    input = {
        "input_images": zip_url,
        "trigger_word": flux_name,
        "autocaption": True,
        "steps": MAX_TRAIN_STEPS,
        "lora_rank": 16,
        "optimizer": "adamw8bit",
        "batch_size": 1,
        "resolution": "512,768,1024",
        "learning_rate": 0.0004,
        "caption_dropout_rate": 0.05,
        "cache_latents_to_disk": False,
        "gradient_checkpointing": False
        }
    data = {
        "destination": f"{USER_NAME}/{flux_name}_flux_lora",
        "input": input,
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response = response.json()
        st.write(response)
        training_id = response.get("id")
        return training_id
    except Exception as e:
        raise Exception(f"Failed to start training: {e}")
    

def start_replicate_training_python(zip_url:str, flux_name:str) -> str:
    # model_name is combinination of model_owner and model_name
    training = replicate.trainings.create(
        model=MODEL_NAME,
        version=MODEL_VERSION,
        input={
        "input_images": zip_url,
        "token_string": flux_name,
        "max_train_steps": MAX_TRAIN_STEPS
        },
        destination=f"{USER_NAME}/{flux_name}_flux_lora"

    )
    return training.id
    
def check_training_status(training_id) -> dict:
    while True:
        url = f"https://api.replicate.com/v1/trainings/{training_id}"
        headers = {
            "Authorization": f"Bearer {REPLICATE_API_KEY}"
        }
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


st.set_page_config(page_title="Flux LoRA Trainer", page_icon="🚀")
st.title("Flux LoRA Trainer")
st.write("Upload a ZIP file containing images to train a Flux LoRA model.")
st.write("Expected File Name: `******_firstnamelastname.zip`")
uploaded_file = st.file_uploader("Drag and drop a .zip file", type=["zip"])

if uploaded_file:
    filename = uploaded_file.name
    if not filename.endswith(".zip"):
        st.error("Please upload a ZIP file.")
        st.stop()
    flux_name = extract_flux_name(filename)
    with st.spinner("Uploading file to Replicate Files API..."):
        zip_url = create_zip_file_link(uploaded_file)
        st.success(f"File uploaded successfully!")

    with st.spinner("Initializing training..."):
        create_destination_model(flux_name=flux_name)
        training_id = start_replicate_training_python(zip_url, flux_name)
        print(training_id)

    if training_id:
        st.success(f"Training started! Training ID: `{training_id}`")
        with st.spinner("Training in progress... This may take a while."):
            time.sleep(5)
            training_status = check_training_status(training_id)

        if training_status.get("status") == "succeeded":
            st.success(f"🎉 Training complete! [View Model]({training_status.destination})")
        else:
            st.error("Training failed. Please try again.")
            st.stop()

