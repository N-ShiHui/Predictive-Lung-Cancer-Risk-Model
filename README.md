# AIAP-Batch-16
# Name: Ng Shi Hui
# Email address: gsh.bomb@gmail.com
# This folder contains a script which uses GitHub Actions to execute my machine learning(ML) pipeline, a sub-folder containing a python file in .py format constitues the end-to-end ML pipeline, this README.md file which gives an overview of this folder and explanations on the pipeline design and its usage.
# Instructions: To execute this pipeline, you will need to import your data and edit the features required for cleaning, data transformation and features creation in the code. Parameters are to be modified upon implementation of the functions.
# This pipeline first reads the datafile and read it as a dataframe in python. EDA was done using Jupyter Notebook, hence this pipeline was constructed on the account that EDA has been completed. Irrelevant features are filtered to cut-down on extra work done initially, followed by data transformation and feature creation. The data type needs to be changed accordingly to the type of model that is being built. In this case, this pipeline is being used to build supervised machine learning models. Train and test datasets are created next to be of aid in building the ML models. Different machine learning algorithms are then built after this step to find the 'winning' performing model to use. The modules and functions to use will only be imported upon building the ML models so as to save time.

# Key findings from EDA: Patients of higher age, whom are male, with COPD history, without genetic markers, exposed to high levels of air pollution, taken bronchodilators, are right-handers, smokes lesser and gains/loses weight between 2.5 to 10kg have a higher chance of lung cancer occurrence. Most surprisingly, fatigue is not shown for patients with lung cancer occurrence.

# The table in the link shows how the features are being processed
![image](https://github.com/N-ShiHui/AIAP-Batch-16/assets/161719020/37df5e9c-26bd-4992-bc26-b727a882e613)
# My choice of models for each machine learning task depends on the type of data, its size and suitability and the models' performance. The 'winning' model will only emerge after evaluation with metrics such as the auc-roc and precision values.
# Upon executing the pipeline in python, the data itself seems to have issues and hence NaN values were unable to be identified even though numerous checks were done in python to ensure there were no NaN values in the input.
# There're other considerations for deploying the models developed such as the accuracy of the models' performance results as well as the time and costs required to run the developed models.
