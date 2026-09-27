import catboost
import numpy as np
import pandas as pd
import optuna, utils, transform

from utils import cat_feats


class Trainer():
    def __init__(self, **params):
        self.params = params
        self.model = catboost.CatBoostRegressor



    def fit(self, x_train: pd.DataFrame, y_train: np.ndarray, x_valid: pd.DataFrame=None, y_valid: np.ndarray=None):
        self.model = self.model(cat_features=utils.cat_feats, **self.params)
        self.model.fit(x_train, y_train)
        return self

    def predict(self, x: pd.DataFrame, y: np.ndarray):
        return self.model.predict(x, y)

    def hyper(self, x_train: pd.DataFrame, y_train: np.ndarray=None, x_valid: pd.DataFrame=None, y_valid: np.ndarray=None):


        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
                'depth': trial.suggest_int('depth', 3, 15),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0),
                'iterations': trial.suggest_int('iterations', 50, 200),
                'random_strength': trial.suggest_float('random_strength', 1e-2, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
                'rsm': trial.suggest_float('rsm', 0.1, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 20),
                'verbose': False,
                'random_state': 42,
                'cat_features': utils.cat_feats
            }

            model = catboost.CatBoostRegressor(**params).fit(x_train, y_train)

            y_score = model.predict(x_valid)
            return utils.WAPE(y_score, y_valid)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=15)
        self.params = study.best_params

    def load(self, x: pd.DataFrame=None):
        trans_ = transform.Transformer(lags=False)
        X = trans_.fit(x)

        y = np.zeros(len(X))
        for i in range(len(X)):
            zapros = X.loc[i,]
            if i - 1 >= 0 and X.at[i, 'route'] == X.at[i - 1, 'route']:
                zapros['lag1'] = y[i - 1]
            else:
                if i == 0:
                    zapros['lag1'] = 0
                else:
                    zapros['lag1'] = np.min(y[:i])

            if i - 2 >= 0 and X.at[i, 'route'] == X.at[i - 1, 'route']:
                zapros['lag2'] = y[i - 1]
            else:
                if i == 0:
                    zapros['lag2'] = 0
                else:
                    zapros['lag2'] = np.min(y[:i])

            if i - 3 >= 0 and X.at[i, 'route'] == X.at[i - 3, 'route']:
                zapros['lag3'] = y[i - 3]
            else:
                if i == 0:
                    zapros['lag3'] = 0
                else:
                    zapros['lag3'] = np.min(y[:i])

            if i - 4 >= 0 and X.at[i, 'route'] == X.at[i - 4, 'route']:
                zapros['lag4'] = y[i - 4]
            else:
                if i == 0:
                    zapros['lag4'] = 0
                else:
                    zapros['lag4'] = np.min(y[:i])

            a = self.model.predict(zapros)
            y[i] = max(a, 0)

        y_ = pd.DataFrame(columns=['prediction'], data=y)
        y_['prediction'] = y_['prediction'].astype('int64')
        q = pd.concat([x, y_], axis=1)
        q.to_csv('C:/Users/aslan/ML/dataset/submission.csv', index=False, header=True, sep=';')
        self.model.save_model("C:/Users/aslan/ML/dataset/catboost_model.cbm")

