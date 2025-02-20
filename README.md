# Job Prediction Models

These job prediction models - Random Forest and Linear Regression - were trained on 9838 unique jobs extracted from 25K job positions from January to December 2024

For easier handling, the dataset used was stored in a dataframe "original_positions" before target encoding and was mapped back into the "positions" after target encoding. 

The RF model showed an <b>MSE of 0.03036734417344173</b> compared to the LR model with an </b>MSE of 5.4643048068967646e-30</b>. 

<b>Note:</b> The RF and LR models were first trained with raw position including categorical variables such as company name, location, language, job description of about 25K but it struggled with extracting meaningful evaluations and had around MSE of 2700+ (RF) and 3700+ (LR). Lgmboost were further used to fine-tune the models but the same MSE figured. Hence, the creation of this repository.

## Actual First Top 20 Positions in the Dataset

| Position                      |
|-------------------------------|
| Fullstack Developer           |
| Frontend Developer            |
| Software Engineer             |
| Software Developer            |
| Senior Frontend Developer     |
| Senior Backend Engineer       |
| Senior Software Engineer      |
| DevOps Engineer               |
| PHP Developer                 |
| Backend Developer             |
| Senior QA Engineer            |
| Senior Fullstack Developer    |
| Fullstack Engineer            |
| Android Developer             |
| QA Engineer                   |
| iOS Developer                 |
| Senior Backend Developer      |
| Senior Frontend Engineer      |
| Backend Engineer              |
| Web Developer                 |


## RF Model Prediction:

Random Forest Mean Squared Error: 0.03036734417344173

| Position                                           | Predicted |
|----------------------------------------------------|-----------|
| Software Developer                                 | 285.72    |
| Senior Backend Engineer                            | 219.40    |
| DevOps Engineer                                    | 210.88    |
| Android Developer                                  | 131.58    |
| Senior Frontend Engineer                           | 121.34    |
| Backend Engineer                                   | 116.58    |
| Web Developer                                      | 103.96    |
| Senior Team Lead of Quality Assurance              | 99.17     |
| Project Manager                                    | 73.26     |
| App Developer                                      | 65.98     |
| HR Manager                                         | 61.43     |
| Java Developer                                     | 60.45     |
| Mobile Developer                                   | 60.15     |
| Salesforce CRM Software Engineering Lead           | 48.11     |
| Frontend Web Developer                             | 43.54     |
| Senior Android Developer                           | 42.16     |
| Data Engineer                                      | 40.59     |
| Salesforce Developer                               | 38.07     |
| Software Developer Fullstack                       | 32.87     |
| Lead Software Quality Assurance Engineer           | 32.14     |


## LR Model Prediction:

Linear Regression Mean Squared Error: 5.4643048068967646e-30

| Position                                | Predicted |
|-----------------------------------------|-----------|
| Fullstack Developer                     | 581.0     |
| Software Developer                      | 286.0     |
| PHP Developer                           | 195.0     |
| Senior QA Engineer                      | 148.0     |
| Fullstack Engineer                      | 143.0     |
| QA Engineer                             | 130.0     |
| Senior Frontend Engineer                | 119.0     |
| Web Developer                           | 111.0     |
| Senior PHP Developer                    | 98.0      |
| Frontend Engineer                       | 75.0      |
| Project Manager                         | 72.0      |
| Data Scientist                          | 67.0      |
| Quality Assurance Engineer              | 62.0      |
| Java Developer                          | 60.0      |
| Mobile Developer                        | 59.0      |
| Backend Software Engineer               | 52.0      |
| Fullstack Software Developer            | 50.0      |
| Frontend Web Developer                  | 44.0      |
| Cloud Engineer                          | 43.0      |
| Mechanical Engineer                     | 42.0      |

## Summary:
<b>Common Positions:</b> Both models successfully identified key positions that are in high demand, demonstrating their relevance.

<b>Ranking Differences:</b> The ranking differences highlight the importance of understanding the specific strengths and limitations of each model.

<b>Prediction Counts:</b> The variation in predicted counts suggests that while both models are useful, they may interpret the data differently.

Overall, both models provide valuable insights into job position demands. The differences in their predictions can offer a more comprehensive understanding when combined, helping to identify key positions and their relative importance in the job market.

These train and test datasets were first used for the Linear Regression Model via R Programming. 

[![Button2](https://img.shields.io/badge/Click%20Me-Button2-green)](https://github.com/rnx2024/Linear-Regression-Model-Training-with-R)

The Random Forest Model in R Programming has categorical limits of only 53 so that it wasn't feasible to do this with R. 
