#Mount Google Drive to access files
from google.colab import drive
drive.mount('/content/drive')

#Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Load the CSV file containing soil data
data = pd.read_csv('/content/drive/My Drive/Cropped Images/filenames.csv')

#Assign image filenames to a new column
data['Image_Filenames'] = data['ID'].apply(lambda x: str(x) + '.png')
labels = data['WSA']

#Split the data into training and testing sets
X_train_filenames, X_test_filenames, y_train, y_test = train_test_split(data['Image_Filenames'], labels, test_size=0.2, random_state=42)

#Normalize the target variable
scaler = MinMaxScaler()
y_train_normalized = scaler.fit_transform(np.array(y_train).reshape(-1, 1))
y_test_normalized = scaler.transform(np.array(y_test).reshape(-1, 1))

#Define image dimensions and batch size
img_height = 224
img_width = 224
batch_size = 32

#Image data generator for loading and preprocessing images
datagen = ImageDataGenerator(rescale=1./255)

#Separate data for training and testing
train_data = data[data['Image_Filenames'].isin(X_train_filenames)]
test_data = data[data['Image_Filenames'].isin(X_test_filenames)]

#Load and preprocess images for training
train_generator = datagen.flow_from_dataframe(
    dataframe=train_data,
    directory='/content/drive/My Drive/Cropped Images/Set 01/',
    x_col='Image_Filenames',
    y_col='WSA',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='raw'
)

#Load and preprocess images for testing
test_generator = datagen.flow_from_dataframe(
    dataframe=test_data,
    directory='/content/drive/My Drive/Cropped Images/Set 01/',
    x_col='Image_Filenames',
    y_col='WSA',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='raw'
)

#Load the CSV file again
data = pd.read_csv('/content/drive/My Drive/Cropped Images/Set 01/DataWSA.csv')

#Print the column names
print(data.columns)

#Import necessary Keras libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

#Define the CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1)
])

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)

#Import matplotlib for visualization
import matplotlib.pyplot as plt

#Evaluate the model
loss = model.evaluate(test_generator)

#Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
