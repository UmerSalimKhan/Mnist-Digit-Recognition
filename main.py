from mnist_model import create_model, train_model, evaluate_model, save_model  
from utils import load_mnist_data 
from visualize import plot_training_history, visualize_images, visualize_model # For visualization 

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    visualize_images(x_train, num_images=10) # Visualize first 10 images

    model_name = "cnn" 
    model = create_model(model_name)

    visualize_model(model) # Visualize the model architecture

    trained_model, history = train_model(model, x_train, y_train, x_test, y_test, epochs=10) 
    evaluate_model(trained_model, x_test, y_test)
    save_model(trained_model) 
    print("Model saved!")

    plot_training_history(history)  # Visualize training history