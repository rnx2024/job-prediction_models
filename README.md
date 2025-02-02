# job-prediction_models

These job prediction models - Random Forest and Linear Regression - were trained on more than 9K unique jobs extracted from 25K job positions from January to December 2024

For easier handling, the dataset used was stored in a dataframe "original_positions" before target encoding and was mapped back into the "positions" after target encoding. 

The RF model showed an <b>MSE of 0.03036734417344173</b> compared to the LR model with an </b>MSE of 5.4643048068967646e-30</b>. 

<b>Note:</b> The RF and LR models were first trained with raw position including categorical variables such as company name, location, language, job description of about 25K but it struggled with extracting meaningful evaluations and had around MSE of 2700+ (RF) and 3700+ (LR). Lgmboost were further used to fine-tune the models but the same MSE figured. Hence, the creation of this repository.


Total rows in the dataset: 9838
First top 20 positions in the dataset:
       fullstack Developer
        frontend Developer
         software Engineer
        software Developer
 senior frontend Developer
   senior backend Engineer
  senior software Engineer
           devops Engineer
             php Developer
         backend Developer
        senior qa Engineer
senior fullstack Developer
        fullstack Engineer
         android Developer
               qa Engineer
             ios Developer
  senior backend Developer
  senior frontend Engineer
          backend Engineer
             web Developer

## RF Model Prediction: 
Random Forest Mean Squared Error: 0.03036734417344173
                                      position  predicted
740                         software Developer     285.72
282                    senior backend Engineer     219.40
830                            devops Engineer     210.88
654                          android Developer     131.58
2613                  senior frontend Engineer     121.34
1600                          backend Engineer     116.58
1643                             web Developer     103.96
653      senior team lead of quality assurance      99.17
2578                           project Manager      73.26
2847                             app Developer      65.98
1412                                hr Manager      61.43
2854                            java Developer      60.45
2746                          mobile Developer      60.15
2278  salesforce crm software Engineering lead      48.11
2724                    frontend web Developer      43.54
512                   senior android Developer      42.16
1881                             data Engineer      40.59
201                       salesforce Developer      38.07
123               software Developer fullstack      32.87
1977  lead software quality assurance Engineer      32.14

## LR Model Prediction:
Linear Regression Mean Squared Error: 5.4643048068967646e-30
                          position  predicted
481            fullstack Developer      581.0
341             software Developer      286.0
1991                 php Developer      195.0
924             senior qa Engineer      148.0
1574            fullstack Engineer      143.0
475                    qa Engineer      130.0
1785      senior frontend Engineer      119.0
1070                 web Developer      111.0
660           senior php Developer       98.0
64               frontend Engineer       75.0
1896               project Manager       72.0
2003                data scientist       67.0
12      quality assurance Engineer       62.0
583                 java Developer       60.0
1586              mobile Developer       59.0
45       backend software Engineer       52.0
2565  fullstack software Developer       50.0
2113        frontend web Developer       44.0
1895                cloud Engineer       43.0
608            mechanical Engineer       42.0

These train and test datasets were first used for the Linear Regression Model via R Programming. \n <https://github.com/rnx2024/Linear-Regression-Model-Training-with-R> The Random Forest Model in R Programming has categorical limits of only 53 so that it wasn't feasible to do this with R. 
