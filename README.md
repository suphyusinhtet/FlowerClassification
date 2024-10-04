# FlowerClassification
Classification of flowers using CNN and Transfer Learning
# Table of Contents
1. [Overview](#overview)
2. [Objectives](#objectives)
3. [Dataset Overview](#dataset)
5. [Model-Implemented](#models-implemented)
6. [Training Process](#training-process)
7. [Model Evaluation](#evaluation-metrics)
8. [Results and Insights](#results-and-insights)
9. [Futre Work](#future-work)
10. [Conclusion](#conclusion)
11. [Acknowledgments](#acknowledgments)

# 🌸 Flower Classification Using CNN and Transfer Learning

## Overview🌟

This project focuses on classifying images of flowers using Convolutional Neural Networks (CNN) and Transfer Learning techniques. We work with pre-trained models like **MobileNetV2** and **Xception** to enhance performance and accuracy. The main goal is to develop a model that can identify different flower types with high precision. The project is deployed on [Streamlit Cloud](https://flowerclassification-123.streamlit.app/).

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

You can download the dataset [here]( https://www.kaggle.com/datasets/alxmamaev/flowers-recognition).

The dataset is split into training and validation sets, with 80% of the data used for training and 20% for validation.

## Models Implemented🤖

### 1. Custom CNN Model
A Convolutional Neural Network (CNN) was built from scratch using several **Conv2D** layers followed by **MaxPooling2D**, **Batch Normalization**, and **Dropout** layers to prevent overfitting. This model was compiled and trained with an **Adam optimizer**.

### 2. Transfer Learning Models
We leveraged pre-trained models like:
- **MobileNetV2**
- **Xception**

These models were fine-tuned to adapt to our flower classification task while freezing the lower layers to retain their learned features from the ImageNet dataset.

## Training Process📈

### Preprocessing🛠️
- **Image Scaling**: The pixel values of images were normalized to a range of [0,1] to accelerate training and improve model performance.
- **Caching and Prefetching**: Implemented to speed up data loading and improve GPU utilization.
  
### Training Workflow🚀
- The dataset was split into training and validation sets using **TensorFlow’s image_dataset_from_directory** function.
- **Callbacks** such as **ReduceLROnPlateau** (for adaptive learning rate), **EarlyStopping** (to prevent overfitting), and **GetBestModel** (to retrieve the best weights) were applied.
  
### Loss Function and Metrics⚙️
- **Loss**: `SparseCategoricalCrossentropy`
- **Metrics**: `Accuracy`

## Evaluation Metrics📊

The models were evaluated using the following metrics:
- **Accuracy**: To measure overall model performance.
- **Loss**: To observe training dynamics.
- **Validation Accuracy**: To ensure the model generalizes well.
- **Validation Loss**: To detect overfitting during training.

## Results and Insights📈

- The custom CNN model showed early signs of **overfitting**, which was tackled using **Dropout**, **Batch Normalization**, and **Early Stopping**.
- The **Transfer Learning models** (MobileNetV2 and Xception) performed better, with faster training and improved accuracy, thanks to the pre-trained knowledge they had from the ImageNet dataset.
- Final accuracy: 95% on the test manual image 
- Model training logs and graphs can be found in the notebooks section.

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
    https://github.com/suphyusinhtet/FlowerClassification.git
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Download the dataset and place it in the `/flowers` directory.

### Running the Project
To run the training script, try in colab:
[<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">]

To run the Streamlit app:
```bash
streamlit run streamlit.py
 ```
## Future Work🔮

- **Data Augmentation**: Experiment with augmentation techniques to enhance generalization.
- **Hyperparameter Tuning**: Optimize learning rate, batch size, and other hyperparameters for better performance.
- **Model Optimization**: Focus on optimizing the deployed model for faster and more efficient real-time predictions.

## Conclusion🌟

This project demonstrates the potential of **CNN** and **Transfer Learning** in the field of image classification. The custom CNN model, although overfitting initially, provided valuable insights into handling overfitting issues. **Transfer learning models** like **MobileNetV2** and **Xception** further enhanced the accuracy of classification. Future improvements could make this project more robust and ready for deployment in real-world applications.

## Contributing

Feel free to submit pull requests or open issues if you want to contribute to the project.

## Acknowledgments🙏

- **Kaggle** for providing the **Flower Classification Dataset**.
- **TensorFlow** for its powerful libraries and tools that made this project possible.
- **Streamlit** for the web app deployment.
