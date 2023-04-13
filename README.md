This project is focused on predicting the species of birds based on their images. It uses machine learning techniques to classify bird images into different categories of bird species. The dataset used in this project is the CUB-200-2011 dataset, which contains images of 200 different species of birds.

Requirements
To run this project, you will need the following:

Python 3.x
Jupyter Notebook
NumPy
Pandas
Scikit-learn
Matplotlib
TensorFlow
Keras
You can install the required packages using the following command:


pip install numpy pandas scikit-learn matplotlib tensorflow keras jupyter
Usage
To run the project, you can use the bird_species_prediction.ipynb notebook. The notebook contains all the code and explanations of each step in the process, from loading the dataset to training and testing the model. You can open the notebook using Jupyter Notebook and run each cell to see the output and results.

Model
The model used in this project is a Convolutional Neural Network (CNN) implemented using TensorFlow and Keras. The model consists of 5 convolutional layers and 3 fully connected layers. The final layer is a softmax layer which outputs the probability distribution over the 200 bird species classes.

Results
The model achieved an accuracy of 80.9% on the testing set
