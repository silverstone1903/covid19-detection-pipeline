from PIL import Image
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as preprocess_input_inception
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as preprocess_input_vgg
import numpy as np


def preprocess_func(model_name):
    if model_name == "inception":
        pp = preprocess_input_inception
    elif model_name == "resnet":
        pp = preprocess_input_resnet
    elif model_name == "vgg":
        pp = preprocess_input_vgg
    else:
        print("Model not found!")
    return pp


def process(image_path, model_name, image_size):

    with Image.open(image_path) as img:
        img = img.resize((image_size, image_size), Image.NEAREST)

    pp = preprocess_func(model_name)
    x = np.array(img, dtype="float32")
    x = np.array([x])
    x = pp(x)

    return x


def load_image(image_file):
    img = Image.open(image_file)
    return img
