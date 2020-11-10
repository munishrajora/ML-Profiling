import sys
import pandas as pd
import numpy as np

# metrics for regression
from sklearn import metrics 
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import r2_score as r2
# from sklearn.metrics import mean_absolute_percentage_error as mape

# metrics for classification
from sklearn.metrics import classification_report
# train test split
from sklearn.model_selection import train_test_split


class model_creation:

    def __init__(self, dataframe=None, label=None, model_type=None, test_size=0.3, split_data=[]):
        """

        :param dataframe: Pandas dataframe (default None)
        :param label: str (default None)
                        column name
        :param model_type: str  (default None)
                Values can be 'regression' or 'classification'

        :param test_size: float  (default 0.3)
                        train test split ratio

        :param split_data: list  (default empty)
                        order of the split_data has to be maintained e.g., split_data = [X_train, X_test, y_train, y_test]


        ::Comment
                    dataframe and split_data, at least one of them must be passed
                    If Both are passed the dataframe will get the priority

        """
        self.dataframe = dataframe
        self.label = label
        self.model_type = model_type
        self.test_size = test_size
        self.split_data = split_data  # order of the split_data has to be maintained e.g., split_data = [X_train, X_test, y_train, y_test]

    def check_parameters(self):
        if self.model_type not in ['regression', 'classification']:
            sys.exit(
                "(Value Error): model_type is not available , please pass the model_type 'regression' or 'classification'")

        if not self.dataframe is None:
            if not self.label is None:
                Y = self.dataframe[self.label]
                X = self.dataframe.drop(self.label, axis=1)
                self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=self.test_size,
                                                                                        random_state=42)
                # print("Train data dim :", self.X_train.shape)
                # print("Test  data dim :", self.X_test.shape)

            else:
                sys.exit("(Value Error): label is not available, please pass the coerect label column name")

        elif not self.split_data is None:

            if input(
                    "cross check the order of the split data [X_train, X_test, y_train, y_test]\nif yes press : 'y'") in [
                'y', 'Y']:
                self.X_train, self.X_test, self.Y_train, self.Y_test = self.split_data[0], self.split_data[1], self.split_data[2], \
                                                                       self.split_data[3]
            else:
                sys.exit("(Value Error) : Need to re-order")

        else:
            sys.exit("(Value Error) : dataframe and split_data both are None , atleast one of them must be passed")
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def create_model(self, model_name=None,model_evaluate=True,seed=30):


        """
        model_name : str

             Regression method                |model_name  |
             ---------------------------------| -----------|
             Linear Regression                | 'lr'       |
             Lasso Regression                 | 'lasso'    |
             Ridge Regression                 | 'ridge'    |
             K Neighbors Regressor            | 'knn'      |
             Support Vector Machine           | 'svm'      |
             DecisionTreeRegressor            | 'dt'       |
             RandomForestRegressor            | 'rf'       |
             GradientBoostingRegressor        | 'gbr'      |
             Light Gradient Boosting Machine  | 'lgbm'     |
             AdaBoost Regressor               | 'ada'      |
             CatBoost Regressor               | 'catboost' |


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
    """
        import warnings
        warnings.filterwarnings("ignore")

        self.X_train, self.X_test, self.Y_train, self.Y_test = self.check_parameters()

        if self.model_type == 'regression':

            est_list = ['lr', 'lasso','ridge','svm','knn','dt','rf','gbr','ada','lgbm','catboost']
            if model_name not in est_list:
                sys.exit(
                    '(Value Error): Model_Name Not Available. Please see docstring for list of available estimators.')

            if model_name == "lr":
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()

            elif model_name == 'lasso':
                from sklearn.linear_model import Lasso
                model = Lasso(random_state=seed)

            elif model_name == 'ridge':
                from sklearn.linear_model import Ridge
                model = Ridge(random_state=seed)

            elif model_name == 'svm':
                from sklearn.svm import SVR
                model = SVR()

            elif model_name == 'knn':
                from sklearn.neighbors import KNeighborsRegressor
                model = KNeighborsRegressor()

            elif model_name == 'dt':
                from sklearn.tree import DecisionTreeRegressor
                model = DecisionTreeRegressor(random_state=seed)

            elif model_name == "rf":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(random_state=seed)

            elif model_name == 'gbr':
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(random_state=seed)

            elif model_name == 'ada':
                from sklearn.ensemble import AdaBoostRegressor
                model = AdaBoostRegressor(random_state=seed)

            elif model_name == 'lgbm':
                import lightgbm as lgb
                model = lgb.LGBMRegressor(random_state=seed)

            elif model_name == 'catboost':
                from catboost import CatBoostRegressor
                model = CatBoostRegressor(random_state=seed, silent=True)


            # model.fit(self.X_train, self.Y_train)
            # print("\n", model)
            # # y_true = Y_train
            # self.evaluate(model)

        elif self.model_type == 'classification':

            est_list = ['logreg', 'knn', 'nb', 'svm', 'ridge', 'dt', 'rf', 'ada', 'lda', 'qda', 'gbc', 'lgbm',
                        'catboost']

            if model_name not in est_list:
                sys.exit(
                    '(Value Error): Model_Name Not Available. Please see docstring for list of available estimators.')

            if model_name == "logreg":
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(random_state=30)

            elif model_name == 'knn':
                from sklearn.neighbors import KNeighborsClassifier
                model = KNeighborsClassifier()

            elif model_name == 'nb':
                from sklearn.naive_bayes import GaussianNB
                model = GaussianNB()

            elif model_name == 'svm':
                from sklearn.linear_model import SGDClassifier
                model = SGDClassifier(random_state=30)

            elif model_name == 'ridge':
                from sklearn.linear_model import RidgeClassifier
                model = RidgeClassifier(random_state=30)

            elif model_name == 'dt':
                from sklearn.tree import DecisionTreeClassifier
                model = DecisionTreeClassifier(random_state=30)

            elif model_name == "rf":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(random_state=30)

            elif model_name == 'ada':
                from sklearn.ensemble import AdaBoostClassifier
                model = AdaBoostClassifier(random_state=30)

            elif model_name == 'lda':
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                model = LinearDiscriminantAnalysis()

            elif model_name == 'qda':
                from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
                model = QuadraticDiscriminantAnalysis()

            elif model_name == 'gbc':
                from sklearn.ensemble import GradientBoostingClassifier
                model = GradientBoostingClassifier(random_state=30)

            elif model_name == 'lgbm':
                import lightgbm as lgb
                model = lgb.LGBMClassifier(random_state=30)

            elif model_name == 'catboost':
                from catboost import CatBoostClassifier
                model = CatBoostClassifier(random_state=30,silent=True)

        model.fit(self.X_train, self.Y_train)

        if model_evaluate:
            self.evaluate(model)
        else:
            print("\n", model)

        return model

    def model_with_tuned_param(self, model_name,model_evaluate=True, param_grid=None, fold=3, n_iter=10, cv=3,
                               seed=42):

        """
        model_name : str

             Regression method                |model_name  |
             ---------------------------------| -----------|
             Linear Regression                | 'lr'       |
             Lasso Regression                 | 'lasso'    |
             Ridge Regression                 | 'ridge'    |
             K Neighbors Regressor            | 'knn'      |
             Support Vector Machine           | 'svm'      |
             DecisionTreeRegressor            | 'dt'       |
             RandomForestRegressor            | 'rf'       |
             GradientBoostingRegressor        | 'gbr'      |
             Light Gradient Boosting Machine  | 'lgbm'     |
             AdaBoost Regressor               | 'ada'      |
             CatBoost Regressor               | 'catboost' |


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
             |Gradient Boosting Classifier    |'gbc'      |
             |Light Gradient Boosting Machine |'lgbm'     |
             |CatBoost Classifier             |'catboost' |
    """
        import warnings
        warnings.filterwarnings("ignore")

        self.X_train, self.X_test, self.Y_train, self.Y_test = self.check_parameters()

        from sklearn.model_selection import RandomizedSearchCV

        if self.model_type == 'regression':

            est_list = ['lr', 'lasso', 'ridge', 'svm', 'knn', 'dt', 'rf', 'gbr', 'ada', 'lgbm', 'catboost']
            if model_name not in est_list:
                sys.exit(
                    '(Value Error): Model_Name Not Available. Please see docstring for list of available estimators.')

            if model_name == "lr":
                from sklearn.linear_model import LinearRegression
                if not param_grid:
                    param_grid = {'fit_intercept': [True, False],
                                  'normalize': [True, False]
                                  }
                model_grid = RandomizedSearchCV(estimator=LinearRegression(), param_distributions=param_grid,
                                                n_iter=n_iter, cv=cv, random_state=seed,
                                                n_jobs=-1)

            elif model_name == 'lasso':
                from sklearn.linear_model import Lasso
                if not param_grid:
                    param_grid = {'alpha': np.arange(0, 1, 0.001),
                                  'fit_intercept': [True, False],
                                  'normalize': [True, False],
                                  }
                model_grid = RandomizedSearchCV(estimator=Lasso(random_state=seed),
                                                param_distributions=param_grid, n_iter=n_iter, cv=cv,
                                                random_state=seed, n_jobs=-1)

            elif model_name == 'ridge':
                from sklearn.linear_model import Ridge
                if not param_grid:
                    param_grid = {"alpha": np.arange(0, 1, 0.001),
                                  # [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                  "fit_intercept": [True, False],
                                  "normalize": [True, False],
                                  }
                model_grid = RandomizedSearchCV(estimator=Ridge(random_state=seed), param_distributions=param_grid,
                                                n_iter=n_iter, cv=cv, random_state=seed,
                                                n_jobs=-1)

            elif model_name == 'svm':
                from sklearn.svm import SVR
                if not param_grid:
                    param_grid = {  # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                        # 'float' : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                        'C': np.arange(0, 10, 0.001),  # [0.01, 0.005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        'epsilon': [1.1, 1.2, 1.3, 1.35, 1.4, 1.5, 1.55, 1.6, 1.7, 1.8, 1.9],
                        'shrinking': [True, False]
                    }
                model_grid = RandomizedSearchCV(estimator=SVR(),
                                                param_distributions=param_grid, n_iter=n_iter,
                                                cv=cv, random_state=seed, n_jobs=-1)

            elif model_name == 'knn':
                from sklearn.neighbors import KNeighborsRegressor
                if not param_grid:
                    param_grid = {'n_neighbors': range(1, min(51,self.X_train.shape[0]-2)),
                                  'weights': ['uniform', 'distance'],
                                  'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                                  'leaf_size': [10, 20, 30, 40, 50, 60, 70, 80, 90]
                                  }
                model_grid = RandomizedSearchCV(estimator=KNeighborsRegressor(),
                                                param_distributions=param_grid, n_iter=n_iter,
                                                cv=cv, random_state=seed, n_jobs=-1)


            elif model_name == 'dt':
                from sklearn.tree import DecisionTreeRegressor
                if not param_grid:
                    max_depth = list(set(np.random.randint(1, (max(3, len(self.X_train.columns)) * .85), 20)))
                    max_features = list(set(np.random.randint(1, max(2, len(self.X_train.columns)), 20)))
                    param_grid = {"max_depth": max_depth,
                                  "max_features": max_features,
                                  "min_samples_leaf": [2, 3, 4, 5, 6],
                                  "criterion": ["mse", "mae", "friedman_mse"],
                                  }
                model_grid = RandomizedSearchCV(estimator=DecisionTreeRegressor(random_state=seed),
                                                param_distributions=param_grid, n_iter=n_iter,
                                                cv=cv, random_state=seed, n_jobs=-1)


            elif model_name == "rf":
                # print("\n\tBuilding the RandomForestRegressor :-")
                from sklearn.ensemble import RandomForestRegressor
                if not param_grid:
                    param_grid = {'n_estimators': [10, 20, 30, 40, 50],
                                  'criterion': ['mse', 'mae'],
                                  'max_depth': [2, 3, 4, 5],
                                  'min_samples_split': [2, 5, 7, 9, 10],
                                  'min_samples_leaf': [1, 2, 4, 7, 9],
                                  'max_features': ['auto', 'sqrt', 'log2'],
                                  'bootstrap': [True, False]
                                  }

                model_grid = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=seed),
                                                param_distributions=param_grid,
                                                cv=cv, random_state=seed, n_jobs=-1)

            elif model_name == 'gbr':
                from sklearn.ensemble import GradientBoostingRegressor
                if not param_grid:
                    param_grid = {'loss': ['ls', 'lad', 'huber', 'quantile'],
                                  'n_estimators': [10, 20, 30, 40, 50],
                                  'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                                  'subsample': [0.1, 0.3, 0.5, 0.7, 0.9, 1],
                                  'criterion': ['friedman_mse', 'mse', 'mae'],
                                  'min_samples_split': [2, 4, 5, 7, 9, 10],
                                  'min_samples_leaf': [1, 2, 3, 4, 5, 7],
                                  'max_depth': [2, 3, 4, 5],
                                  'max_features': ['auto', 'sqrt', 'log2']
                                  }
                model_grid = RandomizedSearchCV(estimator=GradientBoostingRegressor(random_state=seed),
                                                param_distributions=param_grid, n_iter=n_iter,
                                                cv=cv, random_state=seed, n_jobs=-1)

            elif model_name == 'ada':
                from sklearn.ensemble import AdaBoostRegressor
                if not param_grid:
                    param_grid = {'n_estimators': np.arange(10, 200, 5),
                                  'learning_rate': np.arange(0, 1, 0.01),
                                  'loss': ["linear", "square", "exponential"]
                                  }
                model_grid = RandomizedSearchCV(estimator=AdaBoostRegressor(random_state=seed),
                                                param_distributions=param_grid,  n_iter=n_iter,
                                                cv=cv, random_state=seed, n_jobs=-1)


            elif model_name == 'lgbm':
                import lightgbm as lgb
                if not param_grid:
                    param_grid = {  # 'boosting_type' : ['gbdt', 'dart', 'goss', 'rf'],
                        'num_leaves': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200],
                        'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
                        'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        'n_estimators': [10, 30, 50, 70, 90, 100, 120, 150, 170, 200],
                        'min_split_gain': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        'reg_alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        'reg_lambda': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    }
                model_grid = RandomizedSearchCV(estimator=lgb.LGBMRegressor(random_state=seed),
                                                param_distributions=param_grid,  n_iter=n_iter,
                                                cv=cv, random_state=seed, n_jobs=-1)

            elif model_name == 'catboost':
                from catboost import CatBoostRegressor
                if not param_grid:
                    param_grid = {'depth': [3, 1, 2, 6, 4, 5, 7, 8, 9, 10],
                                  'iterations': [250, 100, 500, 1000],
                                  'learning_rate': [0.03, 0.001, 0.01, 0.1, 0.2, 0.3],
                                  'l2_leaf_reg': [3, 1, 5, 10, 100],
                                  'border_count': [32, 5, 10, 20, 50, 100, 200],
                                  }
                model_grid = RandomizedSearchCV(estimator=CatBoostRegressor(random_state=seed, silent=True),
                                                param_distributions=param_grid, n_iter=n_iter,
                                                cv=cv, random_state=seed, n_jobs=-1)



            # model_grid.fit(self.X_train, self.Y_train)
            # model = model_grid.best_estimator_
            # print("\n", model)
            # self.evaluate(model)


        elif self.model_type == 'classification':
            est_list = ['logreg','knn' , 'nb', 'svm', 'ridge','dt', 'rf', 'ada', 'lda','qda','gbc','lgbm','catboost']

            if model_name not in est_list:
                sys.exit(
                    '(Value Error): Model_Name Not Available. Please see docstring for list of available estimators.')

            if model_name == 'logreg':
                from sklearn.linear_model import LogisticRegression
                if not param_grid:
                    param_grid = {'C': np.arange(0, 10, 0.001),
                                  "max_iter" : [1000],
                                  #"penalty": ['l2','none'],
                                  "solver": ['newton-cg', 'lbfgs'],  # handle L2 or no penalty
                                  "class_weight": ["balanced", None]
                                  }
                model_grid = RandomizedSearchCV(estimator=LogisticRegression(random_state=seed),
                                                param_distributions=param_grid, n_iter=n_iter, cv=cv,
                                                random_state=seed, n_jobs=-1)

            elif model_name == 'knn':
                from sklearn.neighbors import KNeighborsClassifier
                if not param_grid:
                    param_grid = {'n_neighbors': range(1, min(51,self.X_train.shape[0]-2)),
                                  'weights': ['uniform', 'distance'],
                                  'metric': ["euclidean", "manhattan"]
                                  }
                model_grid = RandomizedSearchCV(estimator=KNeighborsClassifier(), param_distributions=param_grid,
                                                 n_iter=n_iter, cv=cv, random_state=seed,
                                                n_jobs=-1, iid=False)

            elif model_name == 'nb':
                from sklearn.naive_bayes import GaussianNB
                if not param_grid:
                    param_grid = {'var_smoothing': [0.000000001, 0.000000002, 0.000000005, 0.000000008, 0.000000009,
                                                    0.0000001, 0.0000002, 0.0000003, 0.0000005, 0.0000007, 0.0000009,
                                                    0.00001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.009,
                                                    0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.1, 1]
                                  }
                model_grid = RandomizedSearchCV(estimator=GaussianNB(),
                                                param_distributions=param_grid,  n_iter=n_iter,
                                                cv=cv, random_state=seed, n_jobs=-1)

            elif model_name == 'svm':
                from sklearn.linear_model import SGDClassifier
                if not param_grid:
                    param_grid = {'penalty': ['l2', 'l1', 'elasticnet'],
                                  'l1_ratio': np.arange(0, 1, 0.01),
                                  'alpha': [0.0001, 0.001, 0.01, 0.0002, 0.002, 0.02, 0.0005, 0.005, 0.05],
                                  'fit_intercept': [True, False],
                                  'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                                  'eta0': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
                                  }
                model_grid = RandomizedSearchCV(estimator=SGDClassifier(loss='hinge', random_state=seed),
                                                param_distributions=param_grid,  n_iter=n_iter,
                                                cv=cv, random_state=seed, n_jobs=-1)

            elif model_name == 'ridge':
                from sklearn.linear_model import RidgeClassifier
                if not param_grid:
                    param_grid = {'alpha': np.arange(0, 1, 0.001),
                                  'fit_intercept': [True, False],
                                  'normalize': [True, False]
                                  }
                model_grid = RandomizedSearchCV(estimator=RidgeClassifier(random_state=seed),
                                                param_distributions=param_grid,  n_iter=n_iter,
                                                cv=cv, random_state=seed, n_jobs=-1)

            elif model_name == 'dt':
                from sklearn.tree import DecisionTreeClassifier
                if not param_grid:
                    max_depth = list(set(np.random.randint(1, (max(3, len(self.X_train.columns)) * .85), 20)))
                    max_features =  list(set(np.random.randint(1, max(2, len(self.X_train.columns)), 20)))
                    param_grid = {"max_depth": max_depth,
                                  "max_features": max_features,
                                  "min_samples_leaf": [2, 3, 4, 5, 6],
                                  "criterion": ["gini", "entropy"],
                                  }
                model_grid = RandomizedSearchCV(estimator=DecisionTreeClassifier(random_state=seed),
                                                param_distributions=param_grid,
                                                 n_iter=n_iter, cv=cv, random_state=seed, n_jobs=-1)

            elif model_name == 'rf':
                from sklearn.ensemble import RandomForestClassifier
                if not param_grid:
                    param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                  'criterion': ['gini', 'entropy'],
                                  'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
                                  'min_samples_split': [2, 5, 7, 9, 10],
                                  'min_samples_leaf': [1, 2, 4],
                                  'max_features': ['auto', 'sqrt', 'log2'],
                                  'bootstrap': [True, False]
                                  }
                model_grid = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=seed),
                                                param_distributions=param_grid,  n_iter=n_iter,
                                                cv=cv, random_state=seed, n_jobs=-1)

            elif model_name == 'ada':
                from sklearn.ensemble import AdaBoostClassifier
                if not param_grid:
                    param_grid = {'n_estimators': np.arange(10, 200, 5),  # [10, 40, 70, 80, 90, 100, 120, 140, 150],
                                  'learning_rate': np.arange(0, 1, 0.01),  # [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                                  'algorithm': ["SAMME", "SAMME.R"]
                                  }
                model_grid = RandomizedSearchCV(estimator=AdaBoostClassifier(random_state=seed),
                                                param_distributions=param_grid,  n_iter=n_iter,
                                                cv=cv, random_state=seed, n_jobs=-1)

            elif model_name == 'lda':
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                if not param_grid:
                    param_grid = {'solver': ['lsqr', 'eigen'],
                                  'shrinkage': [None, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                                                0.6, 0.7, 0.8, 0.9, 1]
                                  }
                model_grid = RandomizedSearchCV(estimator=LinearDiscriminantAnalysis(),
                                                param_distributions=param_grid,  n_iter=n_iter,
                                                cv=cv, random_state=seed, n_jobs=-1)

            elif model_name == 'qda':
                from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
                if not param_grid:
                    param_grid = {'reg_param': np.arange(0, 1, 0.01),
                                  }
                model_grid = RandomizedSearchCV(estimator=QuadraticDiscriminantAnalysis(),
                                                param_distributions=param_grid,  n_iter=n_iter,
                                                cv=cv, random_state=seed, n_jobs=-1)

            elif model_name == 'gbc':
                from sklearn.ensemble import GradientBoostingClassifier
                if not param_grid:
                    param_grid = {  # 'loss': ['deviance', 'exponential'],
                        'n_estimators': np.arange(10, 200, 5),  # [10, 40, 70, 80, 90, 100, 120, 140, 150],
                        'learning_rate': np.arange(0, 1, 0.01),  # [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                        'subsample': np.arange(0.1, 1, 0.05),  # [0.1,0.3,0.5,0.7,0.9,1],
                        'min_samples_split': [2, 4, 5, 7, 9, 10],
                        'min_samples_leaf': [1, 2, 3, 4, 5],
                        'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
                        'max_features': ['auto', 'sqrt', 'log2']
                    }
                model_grid = RandomizedSearchCV(estimator=GradientBoostingClassifier(random_state=seed),
                                                param_distributions=param_grid,  n_iter=n_iter,
                                                cv=cv, random_state=seed, n_jobs=-1)

            elif model_name == 'lgbm':
                import lightgbm as lgb
                if not param_grid:
                    param_grid = {
                        'num_leaves': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200],
                        'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
                        'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        'n_estimators': [10, 30, 50, 70, 90, 100, 120, 150, 170, 200],
                        'min_split_gain': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        'reg_alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        'reg_lambda': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    }
                model_grid = RandomizedSearchCV(estimator=lgb.LGBMClassifier(random_state=seed),
                                                param_distributions=param_grid,  n_iter=n_iter,
                                                cv=cv, random_state=seed, n_jobs=-1)

            elif model_name == 'catboost':
                from catboost import CatBoostClassifier
                if not param_grid:
                    param_grid = {'depth': [3, 1, 2, 6, 4, 5, 7, 8, 9, 10],
                                  'iterations': [250, 100, 500, 1000],
                                  'learning_rate': [0.03, 0.001, 0.01, 0.1, 0.2, 0.3],
                                  'l2_leaf_reg': [3, 1, 5, 10, 100],
                                  'border_count': [32, 5, 10, 20, 50, 100, 200],
                                  # 'ctr_border_count':[50,5,10,20,100,200]
                                  }

                model_grid = RandomizedSearchCV(estimator=CatBoostClassifier(random_state=seed, silent=True),
                                                param_distributions=param_grid,  n_iter=n_iter,
                                                cv=cv, random_state=seed, n_jobs=-1)


        model_grid.fit(self.X_train, self.Y_train)
        model = model_grid.best_estimator_

        if model_evaluate:
            self.evaluate(model)
        else:
            print("\n", model)

        return model

    # mean absolute percentage error
    def mape(self,actual, prediction):
        mask = actual != 0
        return (np.fabs(actual - prediction)/actual)[mask].mean()


    def evaluate(self, model):

        if self.model_type == 'regression':

            # y_pred_train = model.predict(self.X_train)
            # print("\n{} Regression report on Train Data {}".format('*' * 40, '*' * 40))
            # print("\t\tMean Absolute Error             {:.3f}".format(mae(self.Y_train, y_pred_train)))
            # print("\t\tMean Squared Error              {:.3f}".format(mse(self.Y_train, y_pred_train)))
            # print("\t\tMean Absolute Percentage Error  {:.3f}".format(self.mape(self.Y_train, y_pred_train)))
            # print("\t\tR2 Score                        {:.3f}".format(r2(self.Y_train, y_pred_train)))
            # try:
            #     print("\t\tMean Squared Log Error          {:.3f}".format(msle(self.Y_train, y_pred_train)))
            # except:
            #     pass
            #
            #
            # y_pred_test = model.predict(self.X_test)
            # print("\n{} Regression report on Test Data {}".format('*' * 40, '*' * 40))
            # print("\t\tMean Absolute Error             {:.3f}".format(mae(self.Y_test, y_pred_test)))
            # print("\t\tMean Squared Error              {:.3f}".format(mse(self.Y_test, y_pred_test)))
            # print("\t\tMean Absolute Percentage Error  {:.3f}".format(self.mape(self.Y_test, y_pred_test)))
            # print("\t\tR2 Score                        {:.3f}".format(r2(self.Y_test, y_pred_test)))
            # try:
            #     print("\t\tMean Squared Log Error          {:.3f}".format(msle(self.Y_train, y_pred_train)))
            # except:
            #     pass

            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            print("\n{} Regression report on Train Data and Test Data {}".format('*' * 40, '*' * 40))
            print("\n", model)
            print("\n\t\tMetrics                     Train Data\t\tTest Data")
            print("\t\t","-"*54)
            print("\t\tMean Absolute Error             {:.3f}\t\t{:.3f}".format(mae(self.Y_train, y_pred_train),mae(self.Y_test, y_pred_test)))
            print("\t\tMean Squared Error              {:.3f}\t\t{:.3f}".format(mse(self.Y_train, y_pred_train),mse(self.Y_test, y_pred_test)))
            print("\t\tMean Absolute Percentage Error  {:.3f}\t\t{:.3f}".format(self.mape(self.Y_train, y_pred_train),self.mape(self.Y_test, y_pred_test)))
            print("\t\tR2 Score                        {:.3f}\t\t{:.3f}".format(r2(self.Y_train, y_pred_train),r2(self.Y_test, y_pred_test)))
            try:
                print("\t\tMean Squared Log Error          {:.3f}\t\t{:.3f}".format(msle(self.Y_train, y_pred_train),msle(self.Y_train, y_pred_train)))
            except:
                pass



        elif self.model_type == 'classification':

            y_pred_train = model.predict(self.X_train)
            print("\n{} classification report on Train Data {}".format('*' * 40, '*' * 40))
            print(classification_report(self.Y_train, y_pred_train))

            y_pred_test = model.predict(self.X_test)
            print("\n{} classification report on Test Data {}".format('*' * 40, '*' * 40))
            print(classification_report(self.Y_test, y_pred_test))


