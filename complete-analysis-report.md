Overview of the Analysis
The primary objective of this analysis was to develop a machine learning model capable of accurately predicting the risk associated with loans. Specifically, we aimed to distinguish between "healthy" loans, which are likely to be repaid, and "high-risk" loans, which have a significant chance of default. This differentiation is crucial for financial institutions to manage risk effectively, allocate resources efficiently, and ultimately, safeguard their financial stability.

The dataset for this analysis comprised various financial metrics derived from loan applications and borrowers' financial histories. The key piece of information we needed to predict was the loan_status, which was categorized as 0 (healthy loan) or 1 (high-risk loan). This binary classification allowed us to focus on identifying patterns and characteristics that differentiate high-risk loans from healthy ones.

The target variable, loan_status, indicates whether a loan is considered healthy (0) or high-risk (1), with value counts initially reflecting an imbalanced distribution favoring healthy loans. The dataset also included several features relevant to assessing loan risk, such as: loan_size, interest_rate, borrower_income, debt_to_income,num_of_accounts, num_of_accounts, derogatory_marks, total_debt. 


Results
Machine Learning Model : Logistic Regression with Original Data
Accuracy: The model demonstrated a high accuracy, suggesting it performs well on the overall dataset.
Precision: For healthy loans (0), precision was nearly perfect, indicating a very low false positive rate. For high-risk loans (1), precision was lower, reflecting some false positives but still at a good level.
Recall: For healthy loans, recall was exceptionally high, showing the model's effectiveness in identifying true negatives. For high-risk loans, recall was also impressive, indicating strong capability in detecting true positives among potential defaults.

Model (Original Data) demonstrated high accuracy, precision, and recall overall. It performed exceptionally well in identifying healthy loans (label 0) but was slightly less precise in identifying high-risk loans (label 1), though it still showed strong recall for these
