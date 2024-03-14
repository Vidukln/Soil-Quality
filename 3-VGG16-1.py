#Mount Google Drive to access files
from google.colab import drive
drive.mount('/content/drive')

#Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

#Load the CSV file containing soil data
data = pd.read_csv('/content/drive/My Drive/Cropped Images/Set 01/DataWSA.csv')

data['Image_Filenames'] = data['ID'].apply(lambda x: str(x) + '.png')
labels = data['WSA']

#Split the data into training and testing sets
X_train_filenames, X_test_filenames, y_train, y_test = train_test_split(data['Image_Filenames'], labels, test_size=0.2, random_state=42)

#Data normalization
scaler = MinMaxScaler()
y_train_normalized = scaler.fit_transform(np.array(y_train).reshape(-1, 1))
y_test_normalized = scaler.transform(np.array(y_test).reshape(-1, 1))

#Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#Adding regression layers
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1)(x)

#Combine base model and custom layers into a new model
model = Model(inputs=base_model.input, outputs=predictions)

#Freeze pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

#Compile the model
model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['accuracy'])

#Image data generator for loading and preprocessing images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

#Load and preprocess images for training
train_data = data[data['Image_Filenames'].isin(X_train_filenames)]
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    directory='/content/drive/My Drive/Cropped Images/Set 01/',
    x_col='Image_Filenames',
    y_col='WSA',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw'
)

#Load and preprocess images for testing
test_data = data[data['Image_Filenames'].isin(X_test_filenames)]
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_data,
    directory='/content/drive/My Drive/Cropped Images/Set 01/',
    x_col='Image_Filenames',
    y_col='WSA',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw'
)

#Train the model
history = model.fit(train_generator, epochs=50, validation_data=test_generator)

import matplotlib.pyplot as plt

# Plot training and validation loss curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Curves')
plt.show()

# Plot training and validation accuracy curves
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy Curves')
plt.show()

import matplotlib.pyplot as plt

# Plot true vs predicted values for training data
plt.scatter(y_train, model.predict(train_generator), color='blue', label='Training Data')
# Plot true vs predicted values for testing data
plt.scatter(y_test, model.predict(test_generator), color='red', label='Testing Data')

plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.legend()
plt.show()
