from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Definir parámetros
batch_size = 32
image_size = (224, 224)
epochs = 10

# Configurar generadores de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    classes=['lapices']
)

validation_generator = validation_datagen.flow_from_directory(
    'validation',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    classes=['lapices']
)

test_generator = test_datagen.flow_from_directory(
    'test',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    classes=['lapices']
)

# Construir el modelo
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
base_model.trainable = False

model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='softmax')  # Ajustar a la cantidad de clases
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Guardar el modelo entrenado
model.save('model')
