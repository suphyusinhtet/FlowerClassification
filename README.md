# FlowerClassification
Classification of flowers using CNN and Transfer Learning
## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model](#model)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview
This project is an image classification system built using own **CNN model** and **MoblieNet and Xception** as the pre-trained model. The goal is to classify flower images into different categories using CNN and transfer learning and compare the results for evaluation.

The project is deployed on [Streamlit](https://flowerclassification-123.streamlit.app/) and the model is hosted on Google Drive for inference. 
Key features:
- Transfer learning using MoblieNet and Xception
- Callbacks: Early Stopping, ReduceLROnPlateau, and GetBestModel
- Caching for improved model load time with `@st.cache_resource`
