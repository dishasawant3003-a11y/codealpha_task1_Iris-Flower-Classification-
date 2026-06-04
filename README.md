# 🌸 Iris Flower Classification using Machine Learning

## Overview

This project is a Machine Learning web application that predicts the species of an Iris flower based on its sepal and petal measurements.

The model is trained using the famous Iris dataset and deployed using Streamlit for an interactive user experience.

---

## Features

* Predicts Iris flower species:

  * Setosa
  * Versicolor
  * Virginica
* Interactive Streamlit interface
* User-friendly sliders for input
* Displays prediction results instantly
* Shows prediction confidence scores (if supported by the model)

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-Learn
* Streamlit

---

## Project Structure

project-folder/

├── app.py

├── iris_dataset.pkl

├── requirements.txt

├── README.md

└── images/

    ├── setosa.jpg

    ├── versicolor.jpg

    └── virginica.jpg

---

## Installation

### Clone the Repository

git clone https://github.com/your-username/iris-flower-classification.git

cd iris-flower-classification

### Install Dependencies

pip install -r requirements.txt

### Run the Application

streamlit run app.py

---

## Dataset

The Iris dataset contains measurements of iris flowers including:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

The target variable is the flower species.

---

## Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Model Training
4. Model Evaluation
5. Model Serialization using Pickle
6. Deployment using Streamlit

---

## Future Improvements

* Add flower images for each prediction
* Deploy on Streamlit Community Cloud
* Add model comparison dashboard
* Improve UI with custom styling

---

## Author

Disha Sawant

Data Science Student | Machine Learning Enthusiast
