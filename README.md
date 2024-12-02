# Age-and-Gender-Prediction-using-CNN
This project uses a **Convolutional Neural Network (CNN)** to predict the **age** and **gender** of individuals from facial images. It leverages the **UTKFace dataset**, which includes images with age and gender labels.

## Features
- **Gender prediction**: Classifies as Male (0) or Female (1)
- **Age prediction**: A regression task predicting the age of the individual.

  ## Dataset
The dataset used in this project is the **UTKFace dataset** which contains over 20,000 images labeled with age, gender, and ethnicity.


## Tools and Libraries
This project uses the following libraries:
- **TensorFlow**: For building and training the CNN model.
- **Keras**: A high-level neural networks API, running on top of TensorFlow.
- **NumPy**: For data manipulation and handling arrays.
- **Pandas**: For data processing.
- **Matplotlib**: For visualizations.

## Model Architecture

The CNN model is structured as follows:

- **Input Layer**: Accepts images of size 200x200 with 3 channels (RGB).
- **Convolutional Layer 1**: 32 filters, kernel size 3x3, activation function `ReLU`, followed by a `MaxPooling` layer with pool size 2x2.
- **Convolutional Layer 2**: 64 filters, kernel size 3x3, activation function `ReLU`, followed by a `MaxPooling` layer with pool size 2x2.
- **Convolutional Layer 3**: 128 filters, kernel size 3x3, activation function `ReLU`, followed by a `MaxPooling` layer with pool size 2x2.
- **Flatten Layer**: Flattens the 3D output of the last convolutional layer to 1D.
- **Fully Connected Layer 1**: 512 neurons with activation function `ReLU`.
- **Fully Connected Layer 2**: 128 neurons with activation function `ReLU`.
- **Output Layer**:
  - For gender prediction: 1 neuron with a `sigmoid` activation function (binary classification).
  - For age prediction: 1 neuron with a `linear` activation function (regression task).


## Model Evaluation Metrics

After training the model, the following evaluation metrics are used:

### Gender Prediction (Classification Task):
- **Accuracy**: 0.85

### Age Prediction (Regression Task):
- **Age Prediction Mean Squared Error**: 75.36

### Instructions for Running the Project

- Open the notebook in Google Colab.

  OR
### For locally
To install all the dependencies from the requirements.txt

-pip install -r requirements.txt

Ensure the datasets are in the correct directory.

Run each cell in the notebook to see the model training, evaluation, and results.
