from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import pickle
from utils import *

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                type=str,
                default="inception",
                help="model name")
args = vars(ap.parse_args())

n_epochs = 5
bs = 32
seed = 2022
model_name = args["model"]
image_size = 299
if model_name != "inception":
    image_size = 224
pp = preprocess_func(model_name)

print("Selected model: {}".format(model_name))
print("Image size: {}".format(image_size))

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    preprocessing_function=pp)


test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=pp)

train = train_datagen.flow_from_directory('data/train',
                                          target_size=(image_size, image_size),
                                          batch_size=bs,
                                          shuffle=True,
                                          class_mode='categorical',
                                          subset='training',
                                          seed=seed)

val = train_datagen.flow_from_directory(
    'data/train', subset='validation', seed=seed)

test = test_datagen.flow_from_directory('data/test',
                                        target_size=(image_size, image_size),
                                        batch_size=bs,
                                        shuffle=True,
                                        class_mode='categorical')


with open('data/{0}_val_files.pkl'.format(model_name), 'wb') as f:
    pickle.dump(val.filenames, f)

print("Train: {}".format(train.n))
print("Val: {}".format(val.n))
print("Test: {}".format(test.n))

n_class = len(train.class_indices.keys())
cls = list(train.class_indices.keys())
print("Number of classes: {}".format(n_class))
print("Classes: {}".format(cls))

model = model_selector(model_name, n_class, image_size)
file_name = model_name+"_model_NoTop_{0}".format(n_class)

trainer(model, train, val, n_epochs, bs, file_name, model_name)
print("Done!")
