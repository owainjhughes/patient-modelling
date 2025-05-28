# ICU Admission Prediction Model for COVID Patients
## Overview
This repository contains the Python code for predicting ICU admission for patients diagnosed with COVID, based on their medical records. The project is divided into three main parts: 
+ Data Cleansing and Visualisation
+ A basic predictive model 
+ An improved PySpark model using classification and clustering techniques.

## Objective
The goal of the models is to accurately predict the ICU admission of COVID patients using machine learning models, based on other medical factors gleaned from their patient records. The project showcases data preprocessing and cleansing, visualization, model building, evaluation, enhanced data cleansing and model improvement.

## Data
In this project there are three datasets:
+ The raw unsanitized data
+ The sanitized data used in the basic model
+ The further sanitized data used in the improved model

## Models 
In this project there are two models:
+ The basic model that uses techniques such as Random Forest
+ The advanced PySpark model that utilized kmeans clustering and subsampling

## How to Run
The models are created in Python Notebooks. You will require an environement which can run Python Notebooks such as Jupyter ot VSCode. Dependancies can be installed using <code>pip install -r requirements.txt</code>. For PySpark you will also need Java JDK 8 or higher and a JAVA_Hme environment variable set.