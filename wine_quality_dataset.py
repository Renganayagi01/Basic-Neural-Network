import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_openml

# Load the Wine Quality Dataset from OpenML
data = fetch_openml(name="wine-quality-white", as_frame=True)
X = data.data  # Features
y = data.target  # Labels

# Convert labels to integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Encode labels as integers (0-6)

# One-hot encode the labels
y = tf.keras.utils.to_categorical(y, num_classes=len(np.unique(y)))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    Dense(32, activation='relu'),  # Hidden layer
    Dense(y.shape[1], activation='softmax')  # Output layer (7 classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Make predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Display predictions and true labels
true_classes = np.argmax(y_test, axis=1)
print(f"Predicted classes: {predicted_classes}")
print(f"True classes: {true_classes}")
