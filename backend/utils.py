from PIL import Image
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as preprocess_input_inception
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as preprocess_input_vgg
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow import keras


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


def model_selector(model_name, n_cls, image_size):

    if model_name == "vgg":
        base_model = VGG16(weights='imagenet',
                                   include_top=False,
                                   input_shape=((image_size, image_size, 3)))
    elif model_name == "resnet":
        base_model = ResNet50(weights='imagenet',
                              include_top=False,
                              input_shape=((image_size, image_size, 3)))
    elif model_name == "inception":
        base_model = InceptionResNetV2(weights='imagenet',
                                       include_top=False,
                                       input_shape=((image_size, image_size, 3)))
    else:
        print("Model not found!")

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(n_cls,
                        activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    optimizer = Adam()
    opt = Adam(lr=3e-4)

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def trainer(model, train, val, n_epochs, bs, file_name, model_name):

    callbacks_list = [
        keras.callbacks.ModelCheckpoint(filepath='models/{0}.h5'.format(file_name),
                                        monitor='val_loss',
                                        save_best_only=True,
                                        verbose=1),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, verbose=1)
    ]

    history = model.fit(
        train,
        batch_size=bs,
        epochs=n_epochs,
        callbacks=callbacks_list,
        validation_data=val,
        verbose=1)

    score = model.evaluate(val, verbose=0)
    print("Model Name: %s" % model_name)
    print('Val loss:', score[0])
    print('Val accuracy:', score[1])


def load_image(image_file):
    img = Image.open(image_file)
    return img
