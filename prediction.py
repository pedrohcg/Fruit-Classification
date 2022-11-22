from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import numpy as np

img_size = 64

def get_image(image):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels = 3)
    image = tf.image.resize(image, [img_size, img_size], method="bilinear")
    plt.imshow(image.numpy()/255)
    image = tf.expand_dims(image, 0)
    return image

def get_type(path):
    return path.split("\\")[-2]

Le = LabelEncoder()

test_path = Path("./dataset/fruits-360/Test")

test_image_paths = list(test_path.glob("*/*"))
test_image_paths = list(map(lambda x : str(x), test_image_paths))
test_types = list(map(lambda x : get_type(x), test_image_paths))

Le.fit_transform(test_types)

modelPath = "./best_model-40.h5"
model = tf.keras.models.load_model(modelPath)

types = Le.transform(test_types)
types = tf.keras.utils.to_categorical(types)

image = get_image("Red_Apple.jpg")

prediction = model.predict(image)
prediction = np.argmax(prediction, axis=1)
print(Le.inverse_transform(prediction)[0])