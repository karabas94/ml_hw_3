import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss, mean_absolute_error
import joblib
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

"""
1 Побудувати модель логістичної регресії із регуляризацією для будь якого набору даних
Обов'язково включити етапи:
-Розбиття датасету на train/test
-препроцесинг даних. (обробка пропусків, пустих значень і тд). Нормалізація фіч.
  Перетоверння категоріальних фіч за допомогою one-hot encoding (якщо застовно до обраного датасету)
-Оцінка помилку генералізації на тестовому датасеті. (обчислити значення функції втрат, а також accuracy).
"""
data = pd.read_csv('diabetes2.csv')

# first five row
print(f'First five row:\n {data.head()}')
print('\n')

# info
print(f'Info:\n{data.info()}')
print('\n')

# describe
print(f'Describe:\n{data.describe()}')
print('\n')

# count space in column
print(f'Count space in column:\n{data.isnull().sum()}')
print('\n')

# max in column
print(f'Max value of column:\n{data.max()}')
print('\n')

# min in column
print(f'Min value of column:\n{data.min()}')
print('\n')

# count of unique in column
print(f'Count of unique values in column:\n{data.nunique()}')
print('\n')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# splitting dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# feature normalization
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

# creating logistic model
model = LogisticRegression(random_state=0, max_iter=1000).fit(X_train_scaler, y_train)

# predict set test result
predict = model.predict(X_test_scaler)
predict_proba = model.predict_proba(X_test_scaler)

# evaluation og generalization error
accuracy = accuracy_score(y_test, predict)
print(f'Accuracy: {accuracy} %')
log_loss_value = log_loss(y_test, predict_proba)
print(f'Log loss: {log_loss_value}')

print('\n------------------------------------------------------------------------------------------------------------')
"""
2 Дослідити вплив регуляризації на точність моделі лінійної регресії
 1 Для цього:
  -обрати довільний датасет (можна з ДЗ2)
  -Розбити датасет на train/test
  -Виконати препроцесінг даних (стандартизація і тд)
  -Порівняти точність моделей із регуляризаціями Lasso & Ridge & Elastic Net. Зробити висновки
 2 Робота з CV датасетом
  -Розбити датасет на train/test/cv
  -За допомогою Ridge Regression (alpha=1) та CV датасету обрати найкращий гіперпараметр d (Polynomial degree) серед d = [1,2,3,4].
  -Обчислити значення функції втрат на тестовому датасеті. Зберегти найкращу модель
"""

data = pd.read_csv('kc_house_data.csv')
# first 5 row
print(data.head())
# info
print(data.info())
# describe
print(data.describe())
# max in column
print(data.max())
# min in column
print(data.min())
# count on unique in column
print(data.nunique())
# count space in column
print(data.isnull().sum)

X = data.iloc[:, 3:15].values
y = data['price'].values

# splitting dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# feature normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# linear regression model
print('Linear Regression')
linear_model = LinearRegression().fit(X_train_scaled, y_train)
linear_predict = linear_model.predict(X_test_scaled)
linear_mae = mean_absolute_error(y_test, linear_predict)
print(f'Mean absolute error: {linear_mae}')
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_predict))
print(f'Root mean squared error: {linear_rmse}')

print('\n')
# ridge model
print('Ridge')
ridge_model = Ridge(alpha=0.001, max_iter=1000).fit(X_train_scaled, y_train)
ridge_predict = ridge_model.predict(X_test_scaled)
ridge_mae = mean_absolute_error(y_test, ridge_predict)
print(f'Mean absolute error: {ridge_mae}')
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_predict))
print(f'Root mean squared error: {ridge_rmse}')

print('\n')
# Lasso
print('Lasso')
lasso_model = Lasso(alpha=0.001, max_iter=1000, tol=0.1).fit(X_train_scaled, y_train)
lasso_predict = lasso_model.predict(X_test_scaled)
lasso_mae = mean_absolute_error(y_test, lasso_predict)
print(f'Mean absolute error: {lasso_mae}')
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_predict))
print(f'Root mean squared error: {lasso_rmse}')

print('\n')
# ElasticNet
print('ElasticNet')
elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=0.1).fit(X_train_scaled, y_train)
elastic_predict = elastic_model.predict(X_test_scaled)
elastic_mae = mean_absolute_error(y_test, elastic_predict)
print(f'Mean absolute error: {elastic_mae}')
elastic_rmse = np.sqrt(mean_squared_error(y_test, elastic_predict))
print(f'Root mean squared error: {elastic_rmse}')

# inference
if linear_mae < ridge_mae and linear_mae < lasso_mae and linear_mae < elastic_mae:
    print('\nLinear Regression better')
elif ridge_mae < linear_mae and ridge_mae < lasso_mae and ridge_mae < elastic_mae:
    print('\nRidge Regression better')
elif lasso_mae < linear_mae and lasso_mae < ridge_mae and lasso_mae < elastic_mae:
    print('\nLasso Regression better')
elif elastic_mae < linear_mae and elastic_mae < ridge_mae and elastic_mae < lasso_mae:
    print('\nElasticNet Regression better')
else:
    print('\nResults equal')

print('\n-----------------------------------------------------------------------------------------------------------')

# CV
X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.4, random_state=42)
X_eval, X_test, y_eval, y_test = train_test_split(X_other, y_other, test_size=0.5, random_state=42)

# feature normalization
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_eval_sc = scaler.transform(X_eval)
X_test_sc = scaler.transform(X_test)

# choose better parameter of degree
eval_rmse_errors = []

degree = [1, 2, 3, 4]
for d in degree:
    poly_converter = PolynomialFeatures(degree=d, include_bias=False)
    poly_features_train = poly_converter.fit_transform(X_train_sc)
    poly_features_eval = poly_converter.transform(X_eval_sc)

    model = Ridge(alpha=1).fit(poly_features_train, y_train)
    predict = model.predict(poly_features_eval)

    rmse = np.sqrt(mean_squared_error(y_eval, predict))

    eval_rmse_errors.append(rmse)
print(eval_rmse_errors, '\n')

optimal_d = degree[np.argmin(np.array(eval_rmse_errors))]

# creating final Ridge model with best parameter
best_poly_converter = PolynomialFeatures(degree=optimal_d, include_bias=False)
best_features_train = best_poly_converter.fit_transform(X_train_sc)
best_poly_features_test = best_poly_converter.fit_transform(X_test_sc)
final_model = Ridge(alpha=1).fit(best_features_train, y_train)
final_predict = final_model.predict(best_poly_features_test)
rmse = np.sqrt(mean_squared_error(y_test, final_predict))
print(f'RMSE: {rmse}\n')

joblib.dump(final_model, 'final_model.joblib')
