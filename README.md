# FlowerClassification
Classification of flowers using CNN and Transfer Learning
# Table of Contents
1. [Overview](#overview-🌟)
2. [Objectives](#objectives-🎯)
3. [Structure](#structure-📝)
4. [Dataset Overview](#dataset-overview-📊)
5. [Data Preparation](#data-preparation-🛠️)
6. [CNN Architecture](#cnn-architecture-🤖)
7. [Transfer Learning Models](#transfer-learning-models-🚀)
8. [Model Evaluation](#model-evaluation-📊)
9. [Results and Insights](#results-and-insights-📈)
10. [Conclusion](#conclusion-🌟)

# 🌸 Flower Classification Using CNN and Transfer Learning

## Overview🌟

This project focuses on classifying images of flowers using Convolutional Neural Networks (CNN) and Transfer Learning techniques. We work with pre-trained models like **MobileNetV2** and **Xception** to enhance performance and accuracy. The main goal is to develop a model that can identify different flower types with high precision.

## Objectives🎯

- Develop a **custom CNN model** for flower classification.
- Use **Transfer Learning** with pre-trained architectures such as **MobileNetV2** and **Xception**.
- Train, fine-tune, and compare the models' performance.
- Prevent overfitting by applying various deep learning techniques like **Batch Normalization**, **Dropout**, and **Early Stopping**.

## Dataset📊

The dataset used is the **Flower Classification Dataset** from Kaggle, consisting of five different flower classes:
- **Daisy**
- **Dandelion**
- **Rose**
- **Sunflower**
- **Tulip**

The dataset is split into training and validation sets, with 80% of the data used for training and 20% for validation.

### Dataset Directory Structure:
/flowers ├── daisy ├── dandelion ├── rose ├── sunflower └── tulip

## Models Implemented🤖

### 1. Custom CNN Model
A Convolutional Neural Network (CNN) was built from scratch using several **Conv2D** layers followed by **MaxPooling2D**, **Batch Normalization**, and **Dropout** layers to prevent overfitting. This model was compiled and trained with an **Adam optimizer**.

### 2. Transfer Learning Models
We leveraged pre-trained models like:
- **MobileNetV2**
- **Xception**

These models were fine-tuned to adapt to our flower classification task while freezing the lower layers to retain their learned features from the ImageNet dataset.

## Training Process 📈

### Data Augmentation and Preprocessing 🛠️
- **Data Augmentation**: Not used to reduce training time (can be implemented for better generalization).
- **Image Scaling**: The pixel values of images were normalized to a range of [0,1] to accelerate training and improve model performance.
- **Caching and Prefetching**: Implemented to speed up data loading and improve GPU utilization.
  
### Training Workflow 🚀
- The dataset was split into training and validation sets using **TensorFlow’s image_dataset_from_directory** function.
- **Callbacks** such as **ReduceLROnPlateau** (for adaptive learning rate), **EarlyStopping** (to prevent overfitting), and **GetBestModel** (to retrieve the best weights) were applied.
  
### Loss Function and Metrics ⚙️
- **Loss**: `SparseCategoricalCrossentropy`
- **Metrics**: `Accuracy`

## Evaluation Metrics 📊

The models were evaluated using the following metrics:
- **Accuracy**: To measure overall model performance.
- **Loss**: To observe training dynamics.
- **Validation Accuracy**: To ensure the model generalizes well.
- **Validation Loss**: To detect overfitting during training.

## Results and Insights 📈

- The custom CNN model showed early signs of **overfitting**, which was tackled using **Dropout**, **Batch Normalization**, and **Early Stopping**.
- The **Transfer Learning models** (MobileNetV2 and Xception) performed better, with faster training and improved accuracy, thanks to the pre-trained knowledge they had from the ImageNet dataset.

## How to Run 🛠️

### Prerequisites
- **Python 3.7+**
- **TensorFlow 2.x**
- **Keras**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/flower-classification.git
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Download the dataset and place it in the `/flowers` directory.

### Running the Project
To run the training script, use the following command:
```bash
python train_model.py
 ```
## Future Work 🔮

- **Data Augmentation**: Experiment with augmentation techniques to enhance generalization.
- **Hyperparameter Tuning**: Optimize learning rate, batch size, and other hyperparameters for better performance.
- **Deploy the Model**: Use tools like **Streamlit** or **Flask** to deploy the model for real-time flower classification.

## Conclusion 🌟

This project demonstrates the potential of **CNN** and **Transfer Learning** in the field of image classification. The custom CNN model, although overfitting initially, provided valuable insights into handling overfitting issues. **Transfer learning models** like **MobileNetV2** and **Xception** further enhanced the accuracy of classification. Future improvements could make this project more robust and ready for deployment in real-world applications.

## Acknowledgments 🙏

- **Kaggle** for providing the **Flower Classification Dataset**.
- **TensorFlow** for its powerful libraries and tools that made this project possible.
