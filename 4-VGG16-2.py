#Mount Google Drive to access files
from google.colab import drive
drive.mount('/content/drive')

#Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Load the CSV file containing soil data
data = pd.read_csv('/content/drive/My Drive/Cropped Images/filenames.csv')

data['Image_Filenames'] = data['Filename'].apply(lambda x: str(x) + '.png')
labels = data['WSA']

#Split the data into training and testing sets
X_train_filenames, X_test_filenames, y_train, y_test = train_test_split(data['Image_Filenames'], labels, test_size=0.2, random_state=42)

#Data normalization
scaler = MinMaxScaler()
y_train_normalized = scaler.fit_transform(np.array(y_train).reshape(-1, 1))
y_test_normalized = scaler.transform(np.array(y_test).reshape(-1, 1))

#Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#Adding more complex regression layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
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

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#Train the model with early stopping
history = model.fit(train_generator, epochs=50, validation_data=test_generator, callbacks=[early_stopping])

import matplotlib.pyplot as plt

#Plot training and validation loss curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Curves')
plt.show()
