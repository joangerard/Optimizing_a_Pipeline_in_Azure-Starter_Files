<!-- #region -->
# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This is a classification problem. The dataset contains personal information for individuals such as age, job, if he/she is married, etc... Also it contains the target variable 'y' that describes if that individual has received a loan from the bank. We seek to predict wheater the bank should give or not the loan to the individual.

The best performant model while running AutoML was EnsembleVoting and for HyperDrive it was Logistic Regression with a regularization strength of 1.15
and 50 iterations.

## Scikit-learn Pipeline


**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

The train.py file does the following:
- Loading the data from a csv that contains 20 features (age to nr.employeed).
- Cleaning the data such as replacing the string married by 1 or 0 otherwise, drop N/A values.
- Split the dataset into training (80%) and testing (20%)
- Fit a logistic regression model using the regularization strength (C) and the number of maximum iterations (max_it).

Then the pipeline consists on:
- Create an experiment.
- Create a computer cluster.
- Define a random parameter sampling.
- Define an early-stop policy.
- Create a HyperDriveConfig.
- Send the experiment within the HyperDriveConfig to the computer cluster.
- Save the best model.

Hyperdrive should search for the best configuration of parameters C and max_it. It uses a random sampling with C drawn from a uniform distribution between 0.1 and 2 and max_it randomly selects between {50,100,150,200}. 
**What are the benefits of the parameter sampler you chose?**

The benefits are that Hyperdrive will try continues values for C and discrete limited values for max_it. Additionally, Hyperdrive is configured to try 30 different configurations and select the ones that has proved to obtain a better accuracy from the classification algorithm. This saves tons of manual testing for data scientists. 

**What are the benefits of the early stopping policy you chose?**
Bandit policy is based on the slack factor and the evaluation interval. It basically helps you avoid running experiments with stagnated or bad accuracy. That saves CPU running time and thus it saves costs. 

## AutoML
AutoML selects the best model for you during a specific time frame. It uses different models. Some of the parameters used are task="classification" to indicate that we want to solve a classifcation problem, primary_metric="accuracy" to indicate that we want to use accuracy as a performance measure, training_dataset=dataset to indicate the data we want to use to train, 5 crossvalidations, and also a computer target to train the models in a computer cluster instead of the local instance. 

## Pipeline comparison
Hyperdrive gives a model with accuracy of 0.916 and AutoML a model with accuracy of 0.918. They do no present a significant difference in terms of accuracy. 
Hyperdrive tries to search the optimal parameters for a single model whereas AzureML try to search for the best model.

## Future work
Well, AzureML alerts about the inbalance of the dataset in the experiment dashboard so techniques to balance the dataset can help to predict correctly the true positives. 

<!-- #endregion -->
