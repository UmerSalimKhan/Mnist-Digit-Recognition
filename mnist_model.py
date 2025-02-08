import tensorflow as tf
from cnn_model import create_cnn_model

def create_model(model_name="cnn"):
    if model_name == "cnn":
        return create_cnn_model()
    elif model_name == 'ann':
        model = tf.keras.models.Sequential([
            # Layer 1
            tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),  # Dropout for regularization

            # Layer 2
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5), # Dropout for regularization

            # Output layer 
            tf.keras.layers.Dense(10, activation='softmax') # Output layer with 10 classes (digits 0-9)
        ])
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    else: 
        raise ValueError("Invalid model name. Choose from 'cnn' and 'ann' from now.")
    
    return model

def train_model(model, x_train, y_train, x_test, y_test, epochs=10):
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
    return model, history

def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

def save_model(model, filepath="mnist_model.keras"):
    model.save(filepath)