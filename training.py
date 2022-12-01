import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3    
from tensorflow.keras import regularizers
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
        dataset = dataset.map(lambda image, type: (data_augmentation(image), type), num_parallel_calls=AUTOTUNE)

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

train_image_path = list(train_path.glob("*/*"))
train_image_path = list(map(lambda x : str(x), train_image_path))
train_image_types = list(map(lambda x: get_type(x), train_image_path))

#definir um tipo para cada fruta
train_image_types = Le.fit_transform(train_image_types)

train_image_types = tf.keras.utils.to_categorical(train_image_types)

trainPath, validation_path, train_type, validation_type = train_test_split(train_image_path, train_image_types)

#resize imagem
img_size = 64
batch_size = 32

resize = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Resizing(img_size, img_size)])

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip(),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.35),
    tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor = (-0.3, -0.2)),
])

#criando um dataset de treino
train_dataset = get_dataset(trainPath, train_type, batch_size)

#image, type = next(iter(train_dataset))

#criando um dataset de validação
validation_dataset = get_dataset(validation_path, validation_type, batch_size, train=False)

#image, type = next(iter(validation_dataset))

backbone = EfficientNetB3(input_shape=(img_size, img_size, 3),
                                    include_top=False)

model = tf.keras.Sequential([
    backbone, 
    tf.keras.layers.Conv2D(filters = 8, kernel_size = (3, 3), activation ='relu', padding ='same',kernel_regularizer = regularizers.l2(l = 0.015),input_shape = (img_size, img_size, 3)),
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters = 16, kernel_size = (3, 3), activation ='relu', kernel_regularizer = regularizers.l2(l = 0.015), padding ='same'),
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), activation ='relu', kernel_regularizer = regularizers.l2(l = 0.015), padding ='same'),
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), activation ='relu',kernel_regularizer = regularizers.l2(l = 0.015),padding ='same'),
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), activation ='relu', kernel_regularizer = regularizers.l2(l = 0.015),padding ='same'),
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,kernel_regularizer = regularizers.l2(l = 0.015), activation='relu'),
    tf.keras.layers.Dropout(rate=.5),
    tf.keras.layers.Dense(131, activation = 'softmax')
    ])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

#salvar o melhor modelo
checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model.h5", verbose=1, save_best_only=True)

#reduzir o lr se muitas gerações passarem sem melhorar
reduceLR = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.8,
    patience=1,
    verbose=1,
    mode="min",
    min_delta=0.000001,
    cooldown=0,
    min_lr=1.0e-08,
)

#para o treinamento se tiver 5 epochs seguidas sem melhora
early_stop = tf.keras.callbacks.EarlyStopping(patience=10)

#treinando o modelo
history = model.fit(
    train_dataset,
    steps_per_epoch = len(trainPath)//batch_size,
    epochs = 40,
    callbacks=[checkpoint, reduceLR, early_stop],
    validation_data = validation_dataset,
    validation_steps = len(validation_path)//batch_size
)

#melhor geração 33