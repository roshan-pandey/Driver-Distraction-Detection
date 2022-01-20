# State Farm Distracted Driver Detection

## 1. Introduction
1.35 million people die in car accidents and most of the accidents happen because of drivers doing multiple tasks while driving. In other words, getting distracted. State Farm collected the data of several subjects in various situations(mentioned below) in a controlled environment.

## 2. Objective

State farm wants to make use of machine learning and deep learning techniques to predict the current state of
the driver. We need to find the likelihood of what the driver is doing in each picture.

## 3. Steps to execute this Project.
NOTE: Running the whole project might take extended period of time. Training time also depends on your systems computation capabilities.
1. Download/fork/clone the project by clicking [here](https://github.com/roshan-pandey/Driver-Distraction-Detection) and place all the data files in ./data/ directory or unzip the project folder provided via NESS and place the imgs directory in ./data/ directory.
2. Go to project folder and use requirements.txt file to install the packages required to run this project. Run this command "pip install -r .\requirements.txt"
3. Go to ./src/modelling and run the model.py file. This will perform all the data manipulation and training of models and save the newly created files in ./data/ directory and models in ./models/model*/ directory.
4. Go to ./notebooks/ directory and run the CSC8635_Roshan_Pandey_210113925.ipynb to generate all the plots again or you can simply see the model comparision performed and the results.

## 4. Reports
There are Two pdf report files:
1. CSC8635_Roshan_Pandey_210113925.pdf: This report was created via jupyter notebook. It contains all the work done and results from this analysis.
3. CSC8635_Roshan_Pandey_Abstract_Report.pdf: This file contains the structured abstract (Context, Objective, Method, Results, Novelty).
4. Git log file can be found in the root directory of the project. ./CSC8635_210113925_gitLog.txt

## 5. Important Consideration
This project runs perfectly fine on windows machine by following the steps mentioned in section 3. If want to run on different OS, encoding might needs to be changed for text processing.
