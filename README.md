# ML-Profiling
ML-profile help us to generate the Multiple ML model with two line of code to understand the behavior of the Machine Learning model
#### It supports classification and regression both with tabular Dataset 
#### functionality :
              1. create_model(dataframe, label, model_type="classification") #without hypyer parameter tuning
              2. model_with_tuned_param(dataframe, label, model_type="classification") #hyper parameter tuning with RandomSearchCV
              

#### list of Algorithm 

##### Regression

| Regression method               |model_name  |
|---------------------------------| -----------|
|Linear Regression                | 'lr'       |
|Lasso Regression                 | 'lasso'    |
|Ridge Regression                 | 'ridge'    |
|K Neighbors Regressor            | 'knn'      |
|Support Vector Machine           | 'svm'      |
|DecisionTreeRegressor            | 'dt'       |
|RandomForestRegressor            | 'rf'       |
|GradientBoostingRegressor        | 'gbr'      |
|Light Gradient Boosting Machine  | 'lgbm'     |
|AdaBoost Regressor               | 'ada'      |
|CatBoost Regressor               | 'catboost' |


##### Classification

|classification method           | model_name|
|--------------------------------|-----------|
|Logistic Regression             |'logreg'   |
|Naive Bayes                     |'nb'       |
|Ridge Classifier                |'ridge'    |
|Random Forest Classifier        |'rf'       |
|Decision Tree Classifier        |'dt'       |
|Linear Discriminant Analysis    |'lda'      |
|Quadratic Discriminant Analysis |'qda'      |
|K Neighbors Classifier          |'knn'      |
|SVM                             |'svm'      |
|Ada Boost Classifier            |'ada'      |
|Gradient Boosting Classifier    |'gb'       |
|Light Gradient Boosting Machine |'lgbm'     |
|CatBoost Classifier             |'catboost' |

