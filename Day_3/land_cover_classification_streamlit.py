import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tempfile
from PIL import Image
import zipfile
import io
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Land Cover Classification",
    page_icon="🛰️",
    layout="wide"
)

# Title and description
st.title("🛰️ Environmental Monitoring and Land Cover Classification")
st.markdown("Upload satellite images to classify land cover types: Cloudy, Desert, Green Area, and Water")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'history' not in st.session_state:
    st.session_state.history = None

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Dataset Upload", "Model Training", "Prediction", "Model Evaluation"])

def extract_dataset(uploaded_file):
    """Extract uploaded zip file and organize data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract zip file
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Create dataframe
        data = pd.DataFrame(columns=['image_path', 'label'])
        
        # Define the labels/classes (adjust paths as needed)
        labels = {
            "cloudy": "Cloudy",
            "desert": "Desert", 
            "green_area": "Green_Area",
            "water": "Water",
        }
        
        # Find the dataset folder
        dataset_path = None
        for root, dirs, files in os.walk(temp_dir):
            for label_folder in labels.keys():
                if label_folder in dirs:
                    dataset_path = root
                    break
            if dataset_path:
                break
        
        if not dataset_path:
            st.error("Could not find the expected folder structure in the uploaded zip file.")
            return None
        
        # Process each folder
        for label_folder, label_name in labels.items():
            folder_path = os.path.join(dataset_path, label_folder)
            if not os.path.exists(folder_path):
                st.warning(f"Folder {label_folder} not found, skipping...")
                continue
                
            # Process each image in the folder
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                if os.path.isfile(image_path):
                    # Copy image to a permanent location or store in session state
                    data = pd.concat([data, pd.DataFrame({
                        'image_path': [image_path], 
                        'label': [label_name]
                    })], ignore_index=True)
        
        return data

def display_sample_images(data):
    """Display sample images from each category"""
    st.subheader("Sample Images from Dataset")
    
    labels = data['label'].unique()
    
    for label in labels:
        st.write(f"**{label}**")
        label_data = data[data['label'] == label]
        
        # Display up to 5 images per row
        cols = st.columns(5)
        sample_images = label_data.sample(min(5, len(label_data)))
        
        for i, (_, row) in enumerate(sample_images.iterrows()):
            try:
                img = Image.open(row['image_path'])
                cols[i].image(img, caption=label, use_column_width=True)
            except Exception as e:
                cols[i].error(f"Error loading image: {e}")

def create_model():
    """Create and return the CNN model"""
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(255, 255, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_confusion_matrix(cm, classes):
    """Plot confusion matrix using matplotlib"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax)
    
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
    return fig

# Page 1: Dataset Upload
if page == "Dataset Upload":
    st.header("📁 Dataset Upload")
    
    uploaded_file = st.file_uploader("Choose a zip file containing the dataset", type="zip")
    
    if uploaded_file is not None:
        with st.spinner("Extracting and processing dataset..."):
            data = extract_dataset(uploaded_file)
            
        if data is not None:
            st.session_state.data = data
            st.success(f"Dataset loaded successfully! Total images: {len(data)}")
            
            # Display dataset statistics
            st.subheader("Dataset Statistics")
            label_counts = data['label'].value_counts()
            st.bar_chart(label_counts)
            
            # Display sample images
            if st.checkbox("Show sample images"):
                display_sample_images(data)

# Page 2: Model Training
elif page == "Model Training":
    st.header("🎯 Model Training")
    
    if st.session_state.data is None:
        st.warning("Please upload a dataset first!")
    else:
        st.write(f"Dataset loaded with {len(st.session_state.data)} images")
        
        # Training parameters
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.slider("Number of epochs", 1, 50, 25)
            batch_size = st.slider("Batch size", 16, 64, 32)
        with col2:
            test_size = st.slider("Test size ratio", 0.1, 0.4, 0.2)
            
        if st.button("Start Training"):
            with st.spinner("Training model..."):
                data = st.session_state.data
                
                # Split data
                train_df, test_df = train_test_split(data, test_size=test_size, random_state=42)
                
                # Data generators
                train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    rotation_range=45,
                    vertical_flip=True,
                    fill_mode='nearest'
                )
                
                test_datagen = ImageDataGenerator(rescale=1./255)
                
                train_generator = train_datagen.flow_from_dataframe(
                    dataframe=train_df,
                    x_col="image_path",
                    y_col="label",
                    target_size=(255, 255),
                    batch_size=batch_size,
                    class_mode="categorical"
                )
                
                test_generator = test_datagen.flow_from_dataframe(
                    dataframe=test_df,
                    x_col="image_path",
                    y_col="label",
                    target_size=(255, 255),
                    batch_size=batch_size,
                    class_mode="categorical"
                )
                
                # Create and train model
                model = create_model()
                
                # Training with progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                class StreamlitCallback:
                    def __init__(self, progress_bar, status_text, epochs):
                        self.progress_bar = progress_bar
                        self.status_text = status_text
                        self.epochs = epochs
                        
                    def on_epoch_end(self, epoch, logs=None):
                        progress = (epoch + 1) / self.epochs
                        self.progress_bar.progress(progress)
                        self.status_text.text(f"Epoch {epoch + 1}/{self.epochs} - Loss: {logs['loss']:.4f} - Accuracy: {logs['accuracy']:.4f}")
                
                # Train the model
                history = model.fit(
                    train_generator,
                    epochs=epochs,
                    validation_data=test_generator,
                    verbose=0
                )
                
                # Save model and history
                st.session_state.model = model
                st.session_state.history = history
                st.session_state.test_generator = test_generator
                
                # Save model to file
                model.save('land_cover_model.h5')
                
                st.success("Model trained successfully!")
                
                # Display training results
                st.subheader("Training Results")
                
                # Plot training history
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Loss plot
                ax1.plot(history.history['loss'], label='Training Loss')
                ax1.plot(history.history['val_loss'], label='Validation Loss')
                ax1.set_title('Training and Validation Loss')
                ax1.set_xlabel('Epochs')
                ax1.set_ylabel('Loss')
                ax1.legend()
                
                # Accuracy plot
                ax2.plot(history.history['accuracy'], label='Training Accuracy')
                ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
                ax2.set_title('Training and Validation Accuracy')
                ax2.set_xlabel('Epochs')
                ax2.set_ylabel('Accuracy')
                ax2.legend()
                
                st.pyplot(fig)
                
                # Final evaluation
                score = model.evaluate(test_generator, verbose=0)
                st.metric("Final Test Accuracy", f"{score[1]:.4f}")

# Page 3: Prediction
elif page == "Prediction":
    st.header("🔍 Single Image Prediction")
    
    if st.session_state.model is None:
        st.warning("Please train a model first!")
    else:
        uploaded_image = st.file_uploader("Choose an image for prediction", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_image is not None:
            # Display the image
            img = Image.open(uploaded_image)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Predict"):
                with st.spinner("Making prediction..."):
                    # Preprocess image
                    img = img.resize((255, 255))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Make prediction
                    prediction = st.session_state.model.predict(img_array)
                    class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']
                    predicted_class = class_names[np.argmax(prediction)]
                    confidence = np.max(prediction)
                    
                    st.success(f"Prediction: {predicted_class}")
                    st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Show prediction probabilities
                    st.subheader("Prediction Probabilities")
                    prob_df = pd.DataFrame({
                        'Class': class_names,
                        'Probability': prediction[0]
                    })
                    st.bar_chart(prob_df.set_index('Class'))

# Page 4: Model Evaluation
elif page == "Model Evaluation":
    st.header("📊 Model Evaluation")
    
    if st.session_state.model is None:
        st.warning("Please train a model first!")
    else:
        if hasattr(st.session_state, 'test_generator'):
            st.subheader("Confusion Matrix")
            
            # Generate predictions
            predictions = st.session_state.model.predict(st.session_state.test_generator)
            actual_labels = st.session_state.test_generator.classes
            predicted_labels = np.argmax(predictions, axis=1)
            
            # Create confusion matrix
            cm = confusion_matrix(actual_labels, predicted_labels)
            class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']
            
            # Plot confusion matrix
            fig = plot_confusion_matrix(cm, class_names)
            st.pyplot(fig)
            
            # Display classification metrics
            st.subheader("Classification Report")
            from sklearn.metrics import classification_report
            report = classification_report(actual_labels, predicted_labels, 
                                         target_names=class_names, output_dict=True)
            
            # Convert to DataFrame for better display
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
        if st.session_state.history is not None:
            st.subheader("Training History")
            
            # Training metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Final Training Accuracy", 
                         f"{st.session_state.history.history['accuracy'][-1]:.4f}")
                st.metric("Final Training Loss", 
                         f"{st.session_state.history.history['loss'][-1]:.4f}")
            
            with col2:
                st.metric("Final Validation Accuracy", 
                         f"{st.session_state.history.history['val_accuracy'][-1]:.4f}")
                st.metric("Final Validation Loss", 
                         f"{st.session_state.history.history['val_loss'][-1]:.4f}")

# File download section
st.sidebar.markdown("---")
st.sidebar.subheader("Download Model")
if st.session_state.model is not None:
    if os.path.exists('land_cover_model.h5'):
        with open('land_cover_model.h5', 'rb') as f:
            st.sidebar.download_button(
                label="Download Trained Model",
                data=f.read(),
                file_name='land_cover_model.h5',
                mime='application/octet-stream'
            )