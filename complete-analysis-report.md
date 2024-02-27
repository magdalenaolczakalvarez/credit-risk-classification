# Module 21: Credit Risk Analysis Report

## Overview of the Analysis

This work aims to employ various techniques to train and evaluate a model's ability to predict the credit risk of borrowers. Historical lending transactions (77,536 data entries) from a peer-to-peer lending services company were used as input into building this model. The csv data imported includes the below column headers, which were used as variables:

Loan size
Interest rate on the loan
Borrower's income
Debt_to_Income ratio
Number of accounts held by the borrower
Derogatory remarks on the borrower's credit report
Total debt
Loan status (with '1' indicating credit default, and '0' representing a performing loan)

The value_counts function reveals that, of the 77,356 loans from the raw data, approximately 3.23% (2,500) are non-performing loans (defaults)

* The stages of the machine learning process for this analysis.

The input data is contained in a CSV file which was read from a 'Resources' folder into a Pandas DataFrame. The data was separated into labels (loan_status) and features (Loan size,Interest rate on the loan,Borrower's income,Debt_to_Income ratio,Number of accounts held by the borrower,Derogatory remarks on the borrower's credit report,and Total debt). The data was then split into training and testing data with the standard 75/25 split. This process was followed by fitting the training data into a Logic Regression Model, and then making predictions based on the testing data, and finally deriving the accuracy score, the confusion matrix, and the classification report.
Using the `RandomOverSampler` module from the imbalanced-learn library the training data was resampled, in contrast with using the previous raw sample data
Another set of accuracy score, confusion matrix, and classification report were derived.

## Results

* Machine Learning Model 1:

According to scikit-learn.org, the accuracy_score function computes the accuracy of correct predictions.The model's accuracy score of approximately 99.18% reveals its reliability in reasonably predicting the credit worthiness of potential borrowers. However, according to Klein (2022), the 'accuracy' measure is not always an adequate performance measure, and the confusion matrix helps make it easy for us to see what kind of confusions occur in our classification algorithms. 
Going by the confusion matrix, we see that the model predicted 18,663 performing loans ('0') correctly, in contrast with the actual of 18,765 (18663 + 102) performing loans, approximately 99% accurate predictions of '0' (performing loans), with a significant precision score of 1. The model also predicted a total of 563 defaults('1') in contrast with the actual of 619 (563 + 56), approximately 91% accurate prediction of '1' (non-performing loans or defaults), with a 0.85 precision score.

* Machine Learning Model 2:

With the resampled data (using the `RandomOverSampler`), the model's accuracy score went up to approximately 99.38% from the previous 99.18%. Also, the model's predictive power of default loans rose to 99% from 91% with a slightly low precision score of 0.84, but relatively remained at approximately 99% predictive power for the healthy (performing) loan, with a precision score of 1.

## Summary

We must consider the tradeoff here when assessing both models. Although we see a significant improvement in the model's accuracy score, and predictive power of default loans using the resampled data, it should be noted that there was also a slight drop in the precision score from 0.85 to 0.84. However, it would be safer to choose a better prediction of defaults because most lenders would want to avoid increase in the number of non-performing loans (defaults).

## References:

https://scikit-learn.org/stable/modules/model_evaluation.html#:~:text=3.3.-,2.2.,function%20returns%20the%20subset%20accuracy.

https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html

Klein, B. (July 5, 2022), Confusion matrix in machine learning, retrieved from python-course.eu: https://python-course.eu/machine-learning/confusion-matrix-in-machine-learning.php
