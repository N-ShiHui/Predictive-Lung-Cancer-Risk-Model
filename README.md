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
##### 5) Split the data according to training(80%), testing(10%) and validation(10%) sets.
##### 6) Train and evaluate baseline models using XGBoost, Catboost, Random Forest and Decision Tree Classifier algorithms.
##### 7) Train and evaluate top performing baseline model after performing hyperparamter tuning.
##### 8) Combine all models and their metrics into dictionaries.
##### 9) Find the best model based on F1 and Accuracy scores.
##### 10) Evaluate the performance of the best model on the test set.

* Key Findings from EDA
##### 1) Majority of the data features are categorical in nature.
##### 2) There is a weak correlation between the features and target feature in this dataset.
##### 3) A change in weight holds the highest correlation with target feature.
##### 4) Patients with lung cancer history experiences higher increase in weight than weight loss.
##### 5) Lung cancer occurs more in Females than Males even though there're slightly more Males than Females in this dataset.
##### 6) Lung cancer occurs more in people with genetic histories in the family line.
##### 7) Dataset consists a higher ratio of elderly patients starting from age 60.
##### 8) Features gender, gene_markers, weight_change, smoking_duration, tiredness_frequency, air_pollution_level and age are highly significant to the target variable upon conducting three-way anova tests.

* Feature handling description
#
|         Feature        |                           Data handling Description                             | 
|------------------------|---------------------------------------------------------------------------------|
| All                    | Column labels are renamed to lower-caps for easier access and readability.      |
| Categorical            | Values are standardized with mapping to replace non-standardized data.          |
| COPD History           | Missing values replaced with 'No'.                                              |
| Taken Bronchodilators  | Missing values replaced with 'No'.                                              |
| Air Pollution Exposure | Missing values replaced with 'Low'.                                             |
| Lung Cancer Occurrence | Data type changed to 'object'.                                                  |
| Start Smoking          | Data type changed to 'int', dropped after feature engineering.                  |
| Stop Smoking           | Data type changed to 'int', dropped after feature engineering.                  |
| Gender                 | Value 'NAN' replaced with nonetype data and dropped.                            |
| ID                     | Deemed as irrelevant and dropped.                                               |
| Age                    | Absolute function applied to remove all negative values.                        |
| weight_change          | Feature engineered with features 'Last Weight' and 'Current Weight'.            |
| smoking_duration       | Feature engineered with features 'Stop Smoking' and 'Start Smoking'.            |
| Last Weight            | Dropped after feature engineering.                                              |
| Current Weight         | Dropped after feature engineering                                               |
##### Below are some code snippets of the data cleaning process
![image](https://github.com/user-attachments/assets/58d54185-5d12-4127-a401-441432aa5244)
![image](https://github.com/user-attachments/assets/95cc9215-38bb-4b0a-acc2-a92bbeea99fa)

* Model selection
|                  Model                  |                    Baseline Model description                       |                   Reason for selecting model                    |
|-----------------------------------------|---------------------------------------------------------------------|-----------------------------------------------------------------|
| Decision Tree Classifier                | Tree-like structure, root node consists of entire dataset.          | Easy to interpret, handles numerical and categorical data.      |
| Categorical Gradient Boosting Classifier| 
| Extreme Gradient Boosting Classifier    |
| Random Forest Classifier                |

