import tensorflow as tf
from tensorflow import keras 
from keras import layers, models
from ultralytics import YOLO  # Assume we're using a YOLOv3 pre-trained model
# from keras.applications import YOLOv3 
import numpy as np

def preprocess_yolo_input(frame):
    """Preprocess each frame for YOLO (resize, normalization, etc.)"""
    # Resize frame to YOLO's expected input size (typically 416x416 or 608x608)
    frame_resized = tf.image.resize(frame, (416, 416))  # Resizing to 416x416 for YOLO
    # Normalize pixel values to [0, 1]
    frame_resized = frame_resized / 255.0
    return frame_resized

def create_conv_lstm_yolo_model(input_shape, yolo_model):
    """
    Create a model combining ConvLSTM for spatio-temporal feature extraction 
    with a pre-trained YOLOv3 for object detection on each frame.
    
    Parameters:
        input_shape (tuple): Input shape for video frames (e.g., (None, height, width, channels))
        yolo_model (tf.keras.Model): Pre-trained YOLO model
    
    Returns:
        model (tf.keras.Model): Final model combining ConvLSTM and YOLO
    """

    input_seq = layers.Input(shape=input_shape)

    # Step 1: ConvLSTM to extract temporal features from the sequence of frames
    x = layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), return_sequences=True, padding='same')(input_seq)
    x = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), return_sequences=True, padding='same')(x)
    
    # Step 2: Apply YOLO on each frame in the sequence
    # YOLO expects individual frames, not sequences. We'll apply YOLO to each frame in the sequence.
    
    # Output list to store predictions for each frame
    frame_predictions = []
    
    batch_size = tf.shape(x)[0]
    sequence_length = input_shape[0]  # Number of frames in the sequence
    
    for i in range(sequence_length):  # Loop over each frame in the sequence
        frame = x[:, i, :, :, :]  # Extract the i-th frame (in batch_size, height, width, channels format)
        
        # Preprocess frame to match YOLO's input format (resize, normalize, etc.)
        frame = preprocess_yolo_input(frame)
        
        # Convert tensor to numpy (this may help with the format issue for YOLO)
        frame_np = frame.numpy()  # Convert tensor to numpy array
        
        # Add batch dimension (YOLO expects a batch of images)
        frame_np = np.expand_dims(frame_np, axis=0)  # Add batch dimension: (1, 416, 416, 3)
        
        # Pass the frame through YOLO (ensure it's a valid input)
        yolo_output = yolo_model(frame_np)  # YOLO expects a batch of images, so pass the numpy array

        frame_predictions.append(yolo_output)
    
    # Step 3: Combine predictions for all frames (can be modified as needed)
    output = layers.concatenate(frame_predictions, axis=1)  # (batch_size, sequence_length, predictions)

    model = models.Model(inputs=input_seq, outputs=output)
    
    return model

# Example usage:
input_shape = (10, 256, 256, 3)  # Sequence of 10 frames of size 256x256x3

yolo_model = YOLO('/Users/diana/Desktop/temp/ultralytics/yolo11n.pt')  # Pre-trained YOLOv3 without top layers

model = create_conv_lstm_yolo_model(input_shape, yolo_model)

# Display model summary to verify
model.summary()
