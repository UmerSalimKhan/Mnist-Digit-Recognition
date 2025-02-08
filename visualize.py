import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model  # Import plot_model

def plot_training_history(history): # Visualize model performance 
    plt.figure(figsize=(12, 5))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    plt.show()

def visualize_images(x_train, num_images=10):  # Function to visualize images
    plt.figure(figsize=(10, 5))  # Figure size 
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)  # Create subplots for images
        plt.imshow(x_train[i].reshape(28, 28), cmap='gray')  # Display image
        plt.title(f"Image {i + 1}")
        plt.axis('off')  # Hide axis labels and ticks
    plt.show()

def visualize_model(model, filename="model_architecture.png"):  # Visualize model architechture
    plot_model(model, to_file=filename, show_shapes=True, show_dtype=True, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96, layer_range=None)
    print(f"Model architecture saved to {filename}")