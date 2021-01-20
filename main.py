#%%
import pandas as pd
import numpy as np

#%%

df_f = pd.read_csv('train_features.csv', parse_dates=['timestamp'])
df_t = pd.read_csv('train_targets.csv', parse_dates=['timestamp'])


#%%

# Заполняем пропуски скользящей средней
def buildDF(df):
    df.fillna(method='ffill', inplace=True)
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['weekday'] = df['timestamp'].dt.dayofweek
    for day in range(1, 31):
        df[f"A_rate_lag_{day}"] = df['A_rate'].shift(day)
        df[f"A_CH4_lag_{day}"] = df['A_CH4'].shift(day)
        df[f"A_C2H6_lag_{day}"] = df['A_C2H6'].shift(day)
        df[f"A_C3H8_lag_{day}"] = df['A_C3H8'].shift(day)
        df[f"A_iC4H10_lag_{day}"] = df['A_iC4H10'].shift(day)
        df[f"A_nC4H10_lag_{day}"] = df['A_nC4H10'].shift(day)
        df[f"A_iC5H12_lag_{day}"] = df['A_iC5H12'].shift(day)
        df[f"A_nC5H12_lag_{day}"] = df['A_nC5H12'].shift(day)
        df[f"A_C6H14_lag_{day}"] = df['A_C6H14'].shift(day)
        df[f"B_rate_lag_{day}"] = df['B_rate'].shift(day)

    df['A_rate_week'] = df['A_rate'].shift(1).rolling(window=7).mean()
    df['A_CH4_mean_week'] = df['A_CH4'].shift(1).rolling(window=7).mean()
    df['A_C2H6_mean_week'] = df['A_C2H6'].shift(1).rolling(window=7).mean()
    df['A_C3H8_mean_week'] = df['A_C3H8'].shift(1).rolling(window=7).mean()
    df['A_iC4H10_mean_week'] = df['A_iC4H10'].shift(1).rolling(window=7).mean()
    df['A_nC4H10_mean_week'] = df['A_nC4H10'].shift(1).rolling(window=7).mean()
    df['A_iC5H12_mean_week'] = df['A_iC5H12'].shift(1).rolling(window=7).mean()
    df['A_nC5H12_mean_week'] = df['A_nC5H12'].shift(1).rolling(window=7).mean()
    df['A_C6H14_mean_week'] = df['A_C6H14'].shift(1).rolling(window=7).mean()
    df['B_rate_mean_week'] = df['B_rate'].shift(1).rolling(window=7).mean()

    df['A_rate_m'] = df['A_rate'].shift(31).rolling(window=30).mean()
    df['A_CH4_mean_m'] = df['A_CH4'].shift(31).rolling(window=30).mean()
    df['A_C2H6_mean_m'] = df['A_C2H6'].shift(31).rolling(window=30).mean()
    df['A_C3H8_mean_m'] = df['A_C3H8'].shift(31).rolling(window=30).mean()
    df['A_iC4H10_mean_m'] = df['A_iC4H10'].shift(31).rolling(window=30).mean()
    df['A_nC4H10_mean_m'] = df['A_nC4H10'].shift(31).rolling(window=30).mean()
    df['A_iC5H12_mean_m'] = df['A_iC5H12'].shift(31).rolling(window=30).mean()
    df['A_nC5H12_mean_m'] = df['A_nC5H12'].shift(31).rolling(window=30).mean()
    df['A_C6H14_mean_m'] = df['A_C6H14'].shift(31).rolling(window=30).mean()
    df['B_rate_mean_m'] = df['B_rate'].shift(31).rolling(window=30).mean()

    df.fillna(method='backfill', inplace=True)

#%%

df_t.fillna(method='pad', inplace=True)
buildDF(df_f)

#%%

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBRegressor

#%%

def TestModel(model, features, target):
    x_train, x_valid, y_train, y_valid = train_test_split(features, target)
    grid = model.fit(x_train, y_train)
    #print(grid.best_params_)
    y_pred = grid.predict(x_valid)
    print("score: ", mean_absolute_error(y_valid, y_pred))
    return grid

#%%

lr = TestModel(LinearRegression(), df_f.drop('timestamp', inplace=False, axis=1), df_t.drop('timestamp', inplace=False, axis=1))

#%%

lasso = TestModel(Lasso(), df_f.drop('timestamp', inplace=False, axis=1), df_t.drop('timestamp', inplace=False, axis=1))

#%%

ridge = TestModel(Ridge(), df_f.drop('timestamp', inplace=False, axis=1), df_t.drop('timestamp', inplace=False, axis=1))

#%%

ada = TestModel(AdaBoostRegressor(random_state=6), df_f.drop('timestamp', inplace=False, axis=1), df_t['B_C2H6'])

#%%

svr = TestModel(SVR(), df_f.drop('timestamp', inplace=False, axis=1), df_t['B_C2H6'])

#%%

lsvr = TestModel(LinearSVR(), df_f.drop('timestamp', inplace=False, axis=1), df_t['B_C2H6'])

#%%

#isotonic = TestModel(IsotonicRegression(), df_f.drop('timestamp', inplace=False, axis=1), df_t.drop('timestamp', inplace=False, axis=1))

#%%

df_test = pd.read_csv('test_features.csv', parse_dates=['timestamp'])
buildDF(df_test)

model_lasso = Lasso(random_state=6).fit(df_f.drop('timestamp', inplace=False, axis=1), df_t.drop('timestamp', inplace=False, axis=1))
df_pred_lasso = model_lasso.predict(df_test.drop('timestamp', inplace=False, axis=1))

model_ridge = Ridge(random_state=6).fit(df_f.drop('timestamp', inplace=False, axis=1), df_t.drop('timestamp', inplace=False, axis=1))
df_pred_ridge = model_ridge.predict(df_test.drop('timestamp', inplace=False, axis=1))

model_linear = LinearRegression().fit(df_f.drop('timestamp', inplace=False, axis=1), df_t.drop('timestamp', inplace=False, axis=1))
df_pred_linear = model_linear.predict(df_test.drop('timestamp', inplace=False, axis=1))

#%%

def stack(models, meta_alg, data_train, targets_train, data_test):
    meta_matrix = np.empty((data_train.shape[0], len(models)))

    for n, model in enumerate(models):
        meta_matrix[:, n] = cross_val_predict(model, data_train, targets_train, cv=5, method='predict')
        model.fit(data_train, targets_train)

    meta_alg.fit(meta_matrix, targets_train)

    meta_matrix_test = np.empty((data_test.shape[0], len(models)))
    for n, model in enumerate(models):
        meta_matrix_test[:, n] = model.predict(data_test)

    return meta_alg.predict(meta_matrix_test)

#%%

#df_pred_B_C2H6 = stack([Lasso(random_state=6), SVR(), AdaBoostRegressor(random_state=6)],
#                XGBRegressor(random_state=6),
#                df_f.drop('timestamp', inplace=False, axis=1),
#                df_t['B_C2H6'],
#                df_test.drop('timestamp', inplace=False, axis=1))


#df_pred_B_iC4H10 = stack([Lasso(random_state=6), SVR(), AdaBoostRegressor(random_state=6)],
#                XGBRegressor(random_state=6),
#                df_f.drop('timestamp', inplace=False, axis=1),
#                df_t['B_iC4H10'],
#                df_test.drop('timestamp', inplace=False, axis=1))


#df_pred_B_C3H8 = stack([Lasso(random_state=6), SVR(), AdaBoostRegressor(random_state=6)],
#                XGBRegressor(random_state=6),
#                df_f.drop('timestamp', inplace=False, axis=1),
#                df_t['B_C3H8'],
#                df_test.drop('timestamp', inplace=False, axis=1))


#df_pred_B_nC4H10 = stack([Lasso(random_state=6), SVR(), AdaBoostRegressor(random_state=6)],
#                XGBRegressor(random_state=6),
#                df_f.drop('timestamp', inplace=False, axis=1),
#                df_t['B_nC4H10'],
#                df_test.drop('timestamp', inplace=False, axis=1))

#%%

df_pred_B_C2H6 = AdaBoostRegressor(Lasso(random_state=6), random_state=6, n_estimators=100).fit(
                df_f.drop('timestamp', inplace=False, axis=1),
                df_t['B_C2H6']).predict(df_test.drop('timestamp', inplace=False, axis=1))


df_pred_B_iC4H10 = AdaBoostRegressor(Lasso(random_state=6), random_state=6, n_estimators=100).fit(
                df_f.drop('timestamp', inplace=False, axis=1),
                df_t['B_iC4H10']).predict(df_test.drop('timestamp', inplace=False, axis=1))


df_pred_B_C3H8 = AdaBoostRegressor(Lasso(random_state=6), random_state=6, n_estimators=100).fit(
                df_f.drop('timestamp', inplace=False, axis=1),
                df_t['B_C3H8']).predict(df_test.drop('timestamp', inplace=False, axis=1))


df_pred_B_nC4H10 = AdaBoostRegressor(Lasso(random_state=6), random_state=6, n_estimators=100).fit(
                df_f.drop('timestamp', inplace=False, axis=1),
                df_t['B_nC4H10']).predict(df_test.drop('timestamp', inplace=False, axis=1))


#%%

df_pred = pd.merge(df_test['timestamp'], pd.DataFrame(df_pred_B_C2H6), left_index=True, right_index=True)
df_pred = pd.merge(df_pred, pd.DataFrame(df_pred_B_C3H8), left_index=True, right_index=True)
df_pred = pd.merge(df_pred, pd.DataFrame(df_pred_B_iC4H10), left_index=True, right_index=True)
df_pred = pd.merge(df_pred, pd.DataFrame(df_pred_B_nC4H10), left_index=True, right_index=True)

df_pred.columns = pd.read_csv('train_targets.csv', parse_dates=['timestamp']).columns

#%%

df_pred.to_csv('pred_targets_ada_only.csv')
