import joblib
import sys
import pandas as pd
import lightgbm as lgb
from yaml import safe_load
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

models = ['xbg', 'rfreg', 'lgb']
TARGET = 'trip_duration'


def load_dataframe(path):
    df = pd.read_csv(path)
    return df
    
    
    
def make_X_y(dataframe:pd.DataFrame,target_column:str):
    df_copy = dataframe.copy()
    
    X = df_copy.drop(columns=target_column)
    y = df_copy[target_column]
    
    return X, y


def train_model(model,X_train,y_train):
    # fit the model on data
    model.fit(X_train,y_train)
    
    return model


def save_model(model,save_path):
    joblib.dump(value=model,
                filename=save_path)
    

def main():
    # current file path
    current_path = Path(__file__)
    # root directory path
    root_path = current_path.parent.parent.parent
    # read input file path
    training_data_path = root_path / sys.argv[1]
    # load the data 
    train_data = load_dataframe(training_data_path)
    # split the data into X and y
    X_train, y_train = make_X_y(dataframe=train_data,target_column=TARGET)
    # read the parameters from params.yaml
    with open('params.yaml') as f:
         params = safe_load(f)
    
    # Looping for different models
    for model in models:
        if model == 'xgb':
            regressor = XGBRegressor()
            file_name = 'xgbreg'
        elif model == 'rfreg':
            file_name = 'rfreg'
            model_params = params['train_model']['random_forest_regressor']
            n_estimators = model_params["n_estimators"]
            max_depth = model_params["max_depth"]
            verbose = model_params["verbose"]
            n_jobs = model_params["n_jobs"]

            regressor = RandomForestRegressor(
                    n_estimators = n_estimators,
                    max_depth = max_depth,
                    verbose = verbose,
                    n_jobs = n_jobs)
        else:
            file_name = 'lightgbmreg'
            model_params = params['train_model']['lightgbm']
            n_estimators = model_params["n_estimators"]
            num_leave = model_params["num_leave"]
            learning_rate = model_params["learning_rate"]
            regressor = lgb.LGBMRegressor(
                    n_estimators = n_estimators,
                    num_leaves = num_leaves,
                    learning_rate = learning_rate)

    
        # train the model
        regressor = train_model(model=regressor,
                                X_train=X_train,
                                y_train=y_train)
        # save the model after training
        model_output_path = root_path / 'models' / 'models'
        model_output_path.mkdir(exist_ok=True)
        save_model(model=regressor,save_path=model_output_path / file_name)
    
    
if __name__ == "__main__":
    main()