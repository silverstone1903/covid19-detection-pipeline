import streamlit as st
from PIL import Image
from utils import load_image, process
import tensorflow as tf
import numpy as np
import gc
import glob
import requests
import datetime as dt
import json


image_size = 224
cls = ['Covid', 'Normal', 'Viral Pneumonia']
label_map_inv = {0: 'Covid', 1: 'Normal', 2: 'Viral Pneumonia'}
label_map = {'Covid': 0, 'Normal': 1, 'Viral Pneumonia': 2}


models = glob.glob("models/*.h5")

st.set_page_config(
    page_title=" COVID-19 detection using Chest X-rays", layout="wide")


page = st.sidebar.selectbox(label="Pages", options=[
    "Main Page", "Predict", "Validation"])

if page == "Main Page":
    st.title('COVID-19 detection using Chest X-rays.')
    st.text('Detect Covid-19 using CV models.')
    st.write(
        "[Dataset Description](https://www.kaggle.com/pranavraikokte/covid19-image-dataset)")


elif page == "Predict":
    st.header("Predict")

    model_list = ["VGG16", "Resnet50", "InceptionResNetV2"]
    model_name = st.selectbox(label="Select your model", options=model_list)

    image_file = st.file_uploader("Upload Images", type=["jpg", "jpeg"])
    if image_file is not None:
        file_details = {"filename": image_file.name, "filetype": image_file.type,
                        "filesize": image_file.size}
        st.image(load_image(image_file),  width=400)

        st.write("Selected model:", model_name)
        with st.spinner('Wait for it...'):
            if st.button("Predict"):
                if model_name == "VGG16":
                    model = tf.keras.models.load_model(
                        [s for s in models if "vgg" in s][0])
                    model_name = "vgg"
                elif model_name == "Resnet50":
                    model = tf.keras.models.load_model(
                        [s for s in models if "resnet" in s][0])
                    model_name = "resnet"
                elif model_name == "InceptionResNetV2":
                    model = tf.keras.models.load_model(
                        [s for s in models if "inception" in s][0])
                    model_name = "inception"
                    image_size = 299
                else:
                    st.error("Please select a model")
                print("{0} model loaded.".format(model_name))

                image = process(image_file, model_name, image_size)
                preds = model.predict(image)
                st.write("Prediction: %s" %
                         (cls[np.argmax(preds)]))
                st.write("Probability:", np.round(
                    preds[0][np.argmax(preds)] * 100, 2))

                del model
                gc.collect()

elif page == "Validation":
    st.header("Validation")
    import pickle
    import random
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    sample_size = st.slider("Sample Size:", value=10,
                            min_value=1, max_value=50)
    model_list = ["VGG16", "Resnet50", "InceptionResNetV2"]
    model_name = st.selectbox(label="Select your model", options=model_list)
    if st.button("Predict"):
        if model_name == "VGG16":
            model = tf.keras.models.load_model(
                [s for s in models if "vgg" in s][0])
            model_name = "vgg"
        elif model_name == "Resnet50":
            model = tf.keras.models.load_model(
                [s for s in models if "resnet" in s][0])
            model_name = "resnet"
        elif model_name == "InceptionResNetV2":
            model = tf.keras.models.load_model(
                [s for s in models if "inception" in s][0])
            model_name = "inception"
            image_size = 299
        else:
            st.error("Please select a model")
        print("Model loaded.")
        st.write("Model loaded")

        with open('data/{0}_val_files.pkl'.format(model_name), 'rb') as f:
            val_files = pickle.load(f)
        val_files = [str("data/train/")+str(c).replace("\\", "/")
                     for c in val_files]
        val_files = [c for c in val_files if c.endswith((".jpg", "jpeg"))]

        val = random.choices(val_files, k=sample_size)
        images, image_names = [], []
        for img in val:
            image_names.append(img)
            image = process(img, model_name, image_size)
            images.append(image)

        truth = [str(c).split("\\")[0] for c in image_names]
        truth = [str(c).split("/")[2] for c in image_names]

        images = np.vstack(images)
        preds_batch = model.predict(images, batch_size=16)
        preds_batch = np.argmax(preds_batch, axis=1)
        del model, images, image_names
        gc.collect()

        st.write(
            "Number of selected random images from validation set: {0}".format(len(val)))
        st.write("Accuracy: ", accuracy_score(
            [*map(label_map.get, truth)], preds_batch))
        st.write("Kappa: ", cohen_kappa_score(
            [*map(label_map.get, truth)], preds_batch))
        st.write("F1 Macro: ", f1_score(
            [*map(label_map.get, truth)], preds_batch, average="macro"))
        st.write("F1 Micro: ", f1_score(
            [*map(label_map.get, truth)], preds_batch, average="micro"))

        scores = {"time": dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                  "model": model_name,
                  "image_count": len(val),
                  "acc": accuracy_score([*map(label_map.get, truth)], preds_batch),
                  "kappa": cohen_kappa_score([*map(label_map.get, truth)], preds_batch),
                  "f1mac": f1_score([*map(label_map.get, truth)], preds_batch, average="macro"),
                  "f1mic": f1_score([*map(label_map.get, truth)], preds_batch, average="micro")}

        response = requests.post(
            url="http://fastapi:8000/addscore", data=json.dumps(scores))
        if response.ok:
            res = "Metrics added to MongoDB"
        else:
            res = "Task failed"
        st.write(res)
