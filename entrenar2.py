import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Rutas a las carpetas de datos
train_data_dir = 'train'
validation_data_dir = 'validation'
test_data_dir = 'test'

# Parámetros
img_width, img_height = 224, 224
batch_size = 10
num_epochs = 10

# Generadores de datos para entrenamiento, validación y prueba
train_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Verificar si hay datos de entrenamiento disponibles
if train_generator.samples > 0:
    # Construir el modelo
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
    ])

    # Configurar el modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size)

    # Guardar el modelo
    model.save('trained_model')

    # Convertir el modelo a TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_saved_model('trained_model')
    tflite_model = converter.convert()

    # Guardar el modelo TensorFlow Lite
    with open('your_model.tflite', 'wb') as f:
        f.write(tflite_model)
else:
    print("No hay datos de entrenamiento disponibles en el directorio 'train'. Verifica la ruta.")
