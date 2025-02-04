# Lung Cancer Prediction with machine learning model.
### This project focuses on analysing the possibility of lung cancer occurrence. Features which holds a higher correlation to the occurrence of lung cancer will be used to build the predictive model. The model will be used to assess if a person is at high risk for lung cancer.

* Prerequisites:
##### Setup environment using python version 3.11.10 and download dependencies stated in requirements.txt

* Pipeline execution
##### 1) In main.py file edit configure path 'config_path' to run config.yaml file in your designated project folder.
##### 2) In config.yaml file edit 'file_path' to read csv file in your designated project folder.
##### 3) Editing of parameters will be made in 'param_grid' if required.
##### 4) If new features are introduced into the dataset, edit 'file_path' accordingly and insert the new features into the feature lists accordingly.

* Logical Flow
##### 1) Load the .yaml file with all the dependencies as well as file path.
##### 2) Read the csv file into a dataframe.
##### 3) Initialize and run the data preparation class built in data_prep.py which consists of preprocessing steps, data has been cleaned beforehand hence it is not required.
##### 4) Initialize the model training class built in model_training.py.
##### 5) 

