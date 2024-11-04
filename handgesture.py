import numpy as np  # For handling numerical data as arrays
import os  # For interacting with the file system
from PIL import Image  # For handling image files
import matplotlib.pyplot as plt  # For plotting graphs and images
from random import randint  # For generating random numbers
import matplotlib.image as mpimg  # For displaying images
from keras import layers, models  # For building neural network layers and models
from keras.utils import to_categorical  # For converting labels to one-hot encoding
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from keras.models import Sequential  # For defining sequential neural network models
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # For defining CNN layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # For evaluation metrics
import seaborn as sns  # For data visualization

# Define a lookup dictionary to label categories and reverse lookup for display
lookup = {}
reverselookup = {}
count = 0

# Populate lookup dictionaries with category labels
for j in os.listdir(r'D:\Pycharm Projects\prodgy_ml_task_03\leapGestRecog\00'):
    if not j.startswith('.'):
        lookup[j] = count
        reverselookup[count] = j
        count += 1
print(lookup)

# Initialize lists for image data and labels
x_data = []
y_data = []
datacount = 0
max_images_per_class = 1000  # Set a limit on the number of images per class

# Load images, resize, and store them along with their labels
for i in range(10):
    for j in os.listdir(r'D:\Pycharm Projects\prodgy_ml_task_03\leapGestRecog\0' + str(i)):
        if not j.startswith('.'):
            count = 0
            for k in os.listdir(r'D:\Pycharm Projects\prodgy_ml_task_03\leapGestRecog\0' + str(i) + '\\' + j):
                if count >= max_images_per_class:  # Enforce max image limit per class
                    break
                # Convert each image to grayscale and resize
                img = Image.open(r'D:\Pycharm Projects\prodgy_ml_task_03\leapGestRecog\0' + str(i) + '\\' + j + '\\' + k).convert('L')
                img = img.resize((80, 40))  # Resize to a smaller resolution
                arr = np.array(img)  # Convert image to numpy array
                x_data.append(arr)  # Append image data
                count += 1
            y_values = np.full((count, 1), lookup[j])  # Assign labels to each image
            y_data.append(y_values)
            datacount += count

# Convert image data and labels to numpy arrays
x_data = np.array(x_data, dtype='float32')
y_data = np.concatenate(y_data, axis=0)

# Reshape image data to fit the input shape for CNN
x_data = x_data.reshape((datacount, 80, 40, 1))
x_data /= 255.0  # Normalize pixel values to [0, 1] range

# Display sample images from each category
for i in range(10):
    plt.imshow(x_data[i * 200, :, :], cmap='gray')
    plt.title(reverselookup[y_data[i * 200, 0]])
    plt.show()

# Convert labels to categorical format for multi-class classification
y_data = to_categorical(y_data)

# Split data into training, validation, and test sets
x_train, x_further, y_train, y_further = train_test_split(x_data, y_data, test_size=0.2)
x_validate, x_test, y_validate, y_test = train_test_split(x_further, y_further, test_size=0.5)

# Define the CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(40, 80, 1)),  # First convolutional layer
    MaxPooling2D(pool_size=(2, 2)),  # First pooling layer
    Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
    MaxPooling2D(pool_size=(2, 2)),  # Second pooling layer
    Conv2D(64, (3, 3), activation='relu'),  # Third convolutional layer
    MaxPooling2D(pool_size=(1, 2)),  # Third pooling layer with adjusted size
    Flatten(),  # Flatten the output for the dense layers
    Dense(128, activation='relu'),  # Fully connected layer
    Dense(10, activation='softmax')  # Output layer with 10 categories
])
# Display the model structure
model.summary()

# Compile the model with specified optimizer and loss function
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model and store the training history
history = model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(x_validate, y_validate))

# Evaluate the model on the test set
[loss, acc] = model.evaluate(x_test, y_test, verbose=1)
print("Accuracy:" + str(acc))

# Plot training and validation accuracy over epochs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot training and validation loss over epochs
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Make predictions on the test data
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get the predicted class labels
y_true = np.argmax(y_test, axis=1)  # Get the true class labels

# Generate and display the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(lookup.keys()))

plt.figure(figsize=(10, 8))
cm_display.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
