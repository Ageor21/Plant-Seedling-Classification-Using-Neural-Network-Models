# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

# Load the data
images = np.load("images.npy")
labels = pd.read_csv("labels.csv")

# 1. Exploratory Data Analysis

# a. Distribution of Classes
print("Shape of images:", images.shape)
print("Shape of labels:", labels.shape)

class_distribution = labels.iloc[:, 0].value_counts()  # Assuming the first column contains the labels
plt.figure(figsize=(10, 6))
class_distribution.plot(kind='bar')
plt.title("Distribution of Classes in the Dataset")
plt.xlabel("Class Labels")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# b. Sample Images with Associated Labels
sample_indices = np.random.choice(images.shape[0], 5, replace=False)
sample_images = images[sample_indices]
sample_labels = labels.iloc[sample_indices, 0]

fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for idx, ax in enumerate(axes):
    ax.imshow(sample_images[idx])
    ax.set_title(sample_labels.iloc[idx])
    ax.axis("off")
plt.tight_layout()
plt.show()

# 2. Data Augmentation

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Visualizing augmented images
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for idx, ax in enumerate(axes):
    augmented_image = datagen.random_transform(sample_images[idx])
    ax.imshow(augmented_image.astype('uint8'))
    ax.set_title(sample_labels.iloc[idx])
    ax.axis("off")
plt.tight_layout()
plt.show()

# 3. Normalize Images
images_normalized = images / 255.0
print("Images normalized. Pixel range is now:", images_normalized.min(), "to", images_normalized.max())

# Visualize a normalized image
plt.imshow(images_normalized[0])
plt.title("Normalized Image")
plt.axis('off')
plt.show()

# Train-Validation-Test Split
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels.iloc[:, 0])

X_train, X_temp, y_train, y_temp = train_test_split(
    images_normalized, encoded_labels, test_size=0.3, random_state=42, stratify=encoded_labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

'''# Save datasets
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
np.save('y_test.npy', y_test)'''

# Define a CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(12, activation='softmax')  # Output layer with 12 classes
])

# Model summary
model.summary()


# 'y_test' is the true labels and 'y_pred' are the predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

model.compile(
    optimizer='adam',  # Optimizer
    loss='sparse_categorical_crossentropy',  # Loss function
    metrics=['accuracy']  # Metrics
)


early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,  
    batch_size=32,
    callbacks=[early_stopping]
)

print(f"Final training epoch: {len(history.history['loss'])}")

train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

print(f"Training Loss: {train_loss}, Training Accuracy: {train_acc}")
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

# Predict on test set
y_test_pred = model.predict(X_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

# Compute accuracy and F1 score
test_accuracy = accuracy_score(y_test, y_test_pred_classes)
test_f1 = f1_score(y_test, y_test_pred_classes, average='weighted')

print(f"Test Accuracy: {test_accuracy}")
print(f"Test F1 Score: {test_f1}")


# Save the trained model
model.save('seedling_classifier_model.h5')

# Code to reload the saved model for inference or retraining
loaded_model = load_model('seedling_classifier_model.h5')
