import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout

# Define image size and batch size
IMG_SIZE = 224
BATCH_SIZE = 32

# Create ImageDataGenerators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
   "C:\\Users\\shami\\OneDrive\\Desktop\\API_Assignment_2\\clothes\\train",
   target_size=(IMG_SIZE, IMG_SIZE),
   batch_size=BATCH_SIZE,
   class_mode='categorical',
   subset='training'
)

validation_generator = train_datagen.flow_from_directory(
   "C:\\Users\\shami\\OneDrive\\Desktop\\API_Assignment_2\\clothes\\train",
   target_size=(IMG_SIZE, IMG_SIZE),
   batch_size=BATCH_SIZE,
   class_mode='categorical',
   subset='validation'
)

# Load the VGG16 model pre-trained on ImageNet, without the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Add custom top layers
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(384, activation='relu')(x)  # Output dimension of 384
output = Dense(6, activation='softmax')(x)  # 6 classes: pants, shirt, shoes, shorts, sneakers, t-shirt

# Create the model
model = Model(inputs=base_model.input, outputs=output)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
   train_generator,
   epochs=50,
   validation_data=validation_generator
)

# Save the model
model.save('garment_classification_model_384d_50.h5')
