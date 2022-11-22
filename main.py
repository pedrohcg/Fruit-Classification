import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3    
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

AUTOTUNE = tf.data.experimental.AUTOTUNE
def get_dataset(path, type, batch_size, train = True):
    image_path = tf.convert_to_tensor(path)
    type = tf.convert_to_tensor(type)

    image_dataset = tf.data.Dataset.from_tensor_slices(image_path)
    types_dataset = tf.data.Dataset.from_tensor_slices(type)

    dataset = tf.data.Dataset.zip((image_dataset, types_dataset))

    dataset = dataset.map(lambda image, type : get_image(image, type))
    dataset = dataset.map(lambda image, type : (resize(image), type), num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)

    if train:
        dataset = dataset.map(lambda image, type: (data_augumentation(image), type), num_parallel_calls=AUTOTUNE)

    dataset = dataset.repeat()
    return dataset

def get_type(path):
    return path.split("\\")[-2]

def get_image(image, type):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)

    return image, type

tf.random.set_seed(4)
Le = LabelEncoder()

train_path = Path("./dataset/fruits-360/Training")
#test_path = Path("./dataset/fruits-360/Test")

train_image_path = list(train_path.glob("*/*"))
train_image_path = list(map(lambda x : str(x), train_image_path))
train_image_types = list(map(lambda x: get_type(x), train_image_path))

#definir um tipo para cada fruta
train_image_types = Le.fit_transform(train_image_types)

train_image_types = tf.keras.utils.to_categorical(train_image_types)

trainPath, validation_path, train_type, validation_type = train_test_split(train_image_path, train_image_types)

#resize imagem
img_size = 64
batch_size = 128

resize = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Resizing(img_size, img_size)])

data_augumentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor = (-0.3, -0.2))
])

#criando um dataset de treino
train_dataset = get_dataset(trainPath, train_type, batch_size)

image, type = next(iter(train_dataset))

#criando um dataset de validação
validation_dataset = get_dataset(validation_path, validation_type, batch_size, train=False)

image, type = next(iter(validation_dataset))

backbone = EfficientNetB3(input_shape=(img_size, img_size, 3),
                                    include_top=False)

model = tf.keras.Sequential([
    backbone,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(131, activation='softmax')
    ])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01,beta_1=0.9,beta_2=0.999,epsilon=1e-07),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

#treinamento inicial do modelo
history = model.fit(
    train_dataset,
    steps_per_epoch = len(trainPath)//batch_size,
    epochs = 1,
    validation_data = validation_dataset,
    validation_steps = len(validation_path)//batch_size
)

#nao treinar a camada inicial
model.layers[0].trainable = False

#salvar o melhor modelo
checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model.h5", verbose=1, save_best_only=True)

#para o treinamento se tiver 5 epochs seguidas sem melhora
#early_stop = tf.keras.callbacks.EarlyStopping(patience=5)

#treinando o modelo
history = model.fit(
    train_dataset,
    steps_per_epoch = len(trainPath)//batch_size,
    epochs = 40,
    callbacks=[checkpoint],
    validation_data = validation_dataset,
    validation_steps = len(validation_path)//batch_size
)


