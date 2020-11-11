# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Table of Contents
- [Problem Statement](##problem)
- [Approaches](##Approaches)
    - [Scikit-learn Pipeline](###scikit-learn)
    - [AutoML](###AutoML)
- [Summary Results](##summary)
- [Pipeline Comparison](##comparison)

## Problem Statement <a name="problem"></a>
The dataset contains text data collected from phone calls to bank in responce to a marketing campaign. It holds information, such as age, marital status, job, education ect. The problem is a classification problem and the aim is to predict whether a client will subscribe to a term deposit, repsresented by variable 'y'. That is, we have two classes; success (class 1) or not (class 0). There are in total 21 features, including the target variable, and 32.950 rows. 

## Approaches
We applied two different methods to the problem. First we used a Sckiti-learn model to fit the data. The hyperparameters of that model were tuned using HyperDrive. Then we applied an AutoML model and compared the result.
 

### Scikit-learn Pipeline <a name="scikit-learn"></a>

As a first task we need to load and prepare the data. The script *train.py* does the required steps, in particular: 

- load data into TabularDatasetFactory dataset, 
- clean data (i.e. handle missing values by dropping them, one-hot encode categorical features and load data into pandas DataFrame),
- split data into features and target, and into train and test datasets,  
- apply a Sckiti-learn model to fit the training data and compute the accuracy for the test data,
- save the model in the folder "./outputs/".

We used the Scikit-Learn **logistic regression** model. It is a predictive analysis and it is used to describe data and to explain the relationship between te target variable "success or not" and the other independent variables, e.g. marital status, job, education university degree, ect. There are two hyperparameters for the logistic regression model: the *inverse of regularization strength (C)* and *max iteration number (max_iter)*. The aim is to tune those hyperparameters using HyperDrive. 

We can tune the hyperparamters using HyperDrive via the *HyperDriveConfig* class. The configuration steps are colleted in the jupyter notebook *udacity-project.ipynb*. Here are the configurations for the HyperDrive class: 

- Hyperparameter space: *RandomParameterSampling* defines a random sampling over the hyperparameter search spaces for *C* and *max_iter*. The advantages here are that it is not so exhaustive and the lack of bias. It is a good first choice. 
- Early termination policy: *BanditPolicy* defines an early termination policy based on slack criteria and a frequency interval for evaluation. Any run that does ot fall within the specified slack factor (or slack amount) of the evaluation metric with respect to the best performing run will be terminated. Since our script reports metrics periodically during the execution, it makes sense to include early termination policy. Moreover, doing so avoids overfitting the training data. For more aggressive savings, we chose the Bandit Policy with a small allowable slack.  
- An estimator that will be called with sampled hyperparameters:  SKLearn creates an estimator for training in Scikit-learn experiments (logistic regression model is importet from Scikit-learn); here we also specify the compute target to be used.
- Primary metric name and goal: The name of the primary metric reported by the experiment runs (*accuracy*) and if we wish to maximize or minimize the primary metric (maximize). 
- Max total runs and max concurrent runs : The maximum total number of runs to create and the maximum number of runs to execute concurrently. Note: the number of concurrent runs is gated on the resources available in the specified compute target. Hence ,we need to ensure that the compute target has the available resources for the desired concurrency.

Next we submit the hyperdrive run to the experiment (i.e. launch an experiment) and show run details with the RunDeatails widget. We collect and save the best model, that is, logistic regression with the tuned hyperparameters which yield the best accuracy score. 


### AutoML
As for the Scikit-learn Pipeline, we need to load and prepare the data. An AutoML run is done via the AutoConfig class. Here are the steps to apply an AutoML model:   

- load the data into a TabularDatasetFactory dataset, 
- prepare the data: here we can use a function from the script train.py to clean the data (as described above), which returns features and target as a pandas DataFrame and Series, respectively,
- concentate features and target into one DataFrame,
- split into train and test datasets (or choose *n_cross_validation* when initializing AutoMLConfig)
- get data in TabularDataset form, 
- initiate AutoMLConfig class*,  
- submit the AutoMLConfig run to the experiment (i.e. launch an experiment) and show run details with the RunDeatails widget 
- collect and safe the best model. 

* To initiate the AutoMLConfig class we need to specify: experiment_timeout_minutes, task, primary_metric, training_data, (validation_data or n_cross_validation), label_column_name, compute_target. 

As mentioned above, we can either pass in a test dataset or choose n_cross_validation for cross-validation. A third option is to do neither both, that is, only to pass a train set and after fitting to apply *predict_proba* method to the test set:

```
automl_config = AutoMLConfig(…)
automl_run = experiment.submit(automl_config …)
best_run, fitted_model = automl_run.get_output()
class_prob = fitted_model.predict_proba(X_test)
```
Every AutoML model has featurization automatically applied. Featurization includes automated feature engineering (when "featurization": 'auto') and scaling and normalization, which then impacts the selected algorithm and its hyperparameter values. We can access this information using the *fitted_model*. Here is the featurization summary of all the input features:


## Summary Results <a name="summary"></a>
We obtained the following accuracy scores:  
- HyperDrive: 0.9073 (Hyperparameters used: C=0.4294, max_iter=130)
- AutoMl: 0.91502 (best model(s): MaxAbsScaler, XGBoostClassifier)

## Pipeline Comparison <a name="comparison"></a>
Both for applying HyperDrive and AutoML we need to create a workspace, initiate an experiment, load the data and clean / prepare it. The difference between the two methods is that HyperDrive requires "more coding", meaning:
- we must have a custom-coded machine learning model, such as logistic regression, otherwise, HyperDrive will not know what model to optimize the parameters for,
- we need to specify the parameter search space, 
- define the sampling method over the search space, 
- specify the primary metric to optimize,
- define an early termination policy 
All those steps are not necessary when applying AutoML, i.e. AutoML does it for us. Another difference is the dataset. 

AutoML has a slighty better accuracy score then HyperDrive. This difference might be because the AutoML run used the XXX models and not logistic regression. 
