from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import tensorflow as tf

img_size = 64
batch_size = 128

def get_image(image, type):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels = 3)
    image = tf.image.resize(image, [img_size, img_size], method="bilinear")
    return image, type

def get_type(path):
    return path.split("\\")[-2]

modelPath = "./best_model.h5"
model = tf.keras.models.load_model(modelPath)

Le = LabelEncoder()

test_path = Path("./dataset/fruits-360/Test")

test_image_paths = list(test_path.glob("*/*"))
test_image_paths = list(map(lambda x : str(x), test_image_paths))
test_types = list(map(lambda x : get_type(x), test_image_paths))

Le.fit_transform(test_types)

test_types = Le.transform(test_types)
test_types = tf.keras.utils.to_categorical(test_types)

test_image_paths = tf.convert_to_tensor(test_image_paths)
test_types = tf.convert_to_tensor(test_types)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices((test_image_paths, test_types))
    .map(get_image)
    .batch(batch_size)
)

loss, acc, prec, rec = model.evaluate(test_dataset)

print("Acuracia: ", acc)
print("Precisao: ", prec)