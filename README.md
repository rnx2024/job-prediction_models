# job-prediction_models
These job prediction models - Random Forest and Linear Regression - were trained on more than 9K unique jobs extracted from 25K job positions from January to December 2024

For easier handling, the dataset used was stored in a dataframe "original_positions" before target encoding and was mapped back into the "positions" after target encoding. 

The RF model showed an MSE of 0.03036734417344173 compared to the LR model with an MSE of 5.4643048068967646e-30. 

Note: The RF and LR models were first trained with raw position including categorical variables such as company name, location, language, job description of about 25K but it struggled with extracting meaningful evaluations and had around MSE of 2700+ (RF) and 3700+ (LR). Lgmboost were further used to fine-tune the models but the same MSE figured. Hence, the creation of this repository.

These train and test datasets were first used for the Linear Regression Model via R Programming. <https://github.com/rnx2024/Linear-Regression-Model-Training-with-R>
The Random Forest Model in R Programming has categorical limits of only 53 so that it wasn't feasible to do this with R. 


