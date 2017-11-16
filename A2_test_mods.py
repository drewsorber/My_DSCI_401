import pandas as pd
import matplotlib.pyplot as plt
import pprint
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer

# Get a list of the categorical features for a given dataframe. 
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))

# Get the indices of the categorical features for a given dataframe. 
def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)
	
df = pd.read_csv('./AmesHousingSetA.csv')

#Assign Dummy Variables
df = pd.get_dummies(df, columns = cat_features(df))

#Delete irrelevant data
del df['PID']

#set features and response variable
features = list(df)
features.remove('SalePrice')
data_x = df[features]
data_y = df['SalePrice']

# Imputing the column means for missing values (strategy=most_frequent) by column (axis=0. axis=1 means by row).
imp = preprocessing.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
data_x = imp.fit_transform(data_x)

#baseline model
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
base_model = linear_model.LinearRegression()

# Fit the model.
base_model.fit(x_train, y_train)

# Make predictions on test data and look at the results.
preds = base_model.predict(x_test)
pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))

print('MSE, MAE, R^2, EVS (Base Model): ' + str([mean_squared_error(y_test, preds), \
							   median_absolute_error(y_test, preds), \
							   r2_score(y_test, preds), \
							   explained_variance_score(y_test, preds)])) 
							   
#dataset B attempt

dfb = pd.read_csv('./AmesHousingSetB.csv')

#Assign Dummy Variables
dfb = pd.get_dummies(dfb, columns = cat_features(dfb))

#Delete irrelevant data
del dfb['PID']

#set features and response variable
features = list(dfb)
features.remove('SalePrice')
data_x = dfb[features]
data_y = dfb['SalePrice']

# Imputing the column means for missing values (strategy=most_frequent) by column (axis=0. axis=1 means by row).
imp = preprocessing.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
data_x = imp.fit_transform(data_x)

test_model = linear_model.LinearRegression()
test_model.fit(data_x, data_y)

preds = test_model.predict(data_x)
pprint.pprint(pd.DataFrame({'Actual':data_y, 'Predicted':preds}))

print('Dataset B  - MSE, MAE, R^2, EVS (Base Model): ' + str([mean_squared_error(data_y, preds), \
							   median_absolute_error(data_y, preds), \
							   r2_score(data_y, preds), \
							   explained_variance_score(data_y, preds)])) 	
							   
# Lasso regression 
base_mod = linear_model.LinearRegression()
base_mod.fit(data_x, data_y)

# Make predictions on test data and look at the results.
preds = base_mod.predict(data_x)
print('R^2: ' + str(r2_score(data_y, preds)))

# Show Lasso regression fits for different alphas.
alphas = [0.01, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
for a in alphas:
	 # Normalizing transforms all variables to number of standard deviations away from mean.
	 lasso_mod = linear_model.Lasso(alpha=a, normalize=True, fit_intercept=True)
	 lasso_mod.fit(data_x, data_y)
	 preds = lasso_mod.predict(data_x)
	 print('R^2 (Lasso Model with alpha=' + str(a) + '): ' + str(r2_score(data_y, preds)))

# Make predictions on test data and look at the results.
preds = lasso_mod.predict(data_x)
pprint.pprint(pd.DataFrame({'Actual':data_y, 'Predicted':preds}))
print('For data_x_std: ' + 'MSE, MAE, R^2, EVS: ' + str([mean_squared_error(data_y, preds), \
							   median_absolute_error(data_y, preds), \
							   r2_score(data_y, preds), \
							   explained_variance_score(data_y, preds)])) 
							   
# top 25%
print('%ile-based feature selection')
selector_f = SelectPercentile(f_regression, percentile=25)
selector_f.fit(data_x, data_y)

# Print the f-scores   
for name, score, pv in zip(list(dfb), selector_f.scores_, selector_f.pvalues_):
	print('F-score, p-value (' + name + '): ' + str(score) + ',  ' + str(pv))

#Create a least squares linear regression model.
model3 = linear_model.LinearRegression()

# Fit the model.
model3.fit(data_x, data_y)

# Make predictions on test data and look at the results.
preds = model3.predict(data_x)
pprint.pprint(pd.DataFrame({'Actual':data_y, 'Predicted':preds}))

 # Make predictions on test data and look at the results.
preds = model3.predict(data_x)
pprint.pprint(pd.DataFrame({'Actual':data_y, 'Predicted':preds}))
print('For data_x_std: ' + 'MSE, MAE, R^2, EVS: ' + str([mean_squared_error(data_y, preds), \
							    median_absolute_error(data_y, preds), \
							    r2_score(data_y, preds), \
							    explained_variance_score(data_y, preds)])) 
								
# 3.4 Use k-best feature selection to build the model (Top 3 Model) -- 'MSE, MAE, R^2, EVS (Top 3 Model): [1672431710, 21299.36, .7532, .7538]

# Create a top-k feature selector based on the F-scores. Get top 25% best features by F-test. -- 'MSE, MAE, R^2, EVS (Top 3 Model): [1672431710. 21299, .7532, .7538]
selector_f = SelectKBest(f_regression, k=3)
selector_f.fit(data_x, data_y)

# Create a least squares linear regression model.
model = linear_model.LinearRegression()

# Fit the model.
model.fit(data_x, data_y)

# # Make predictions on test data and look at the results.
preds = model.predict(data_x)
pprint.pprint(pd.DataFrame({'Actual':data_y, 'Predicted':preds}))

print('MSE, MAE, R^2, EVS (Top 3 Model): ' + \
							    str([mean_squared_error(data_y, preds), \
							    median_absolute_error(data_y, preds), \
							    r2_score(data_y, preds), \
							    explained_variance_score(data_y, preds)])) 	
								
3.5 Use Recursive Feature Elimination with Cross Validation -- 'MSE, MAE, R^2, EVS (RFECV Model): [1015898965, 14210, .8501, .8508]
print('Recursive Feature Elimination with Cross Validation')

selector_f = RFECV(estimator=linear_model.LinearRegression(), \
                   cv=5, scoring=make_scorer(r2_score))
selector_f.fit(data_x, data_y)

# Create a least squares linear regression model.
model5 = linear_model.LinearRegression()

# Fit the model.
model5.fit(data_x, data_y)

# Make predictions on test data and look at the results.
preds = model.predict(data_x)
pprint.pprint(pd.DataFrame({'Actual':data_y, 'Predicted':preds}))

print('MSE, MAE, R^2, EVS (RFECV Model): ' + \
							   str([mean_squared_error(data_y, preds), \
							   median_absolute_error(data_y, preds), \
							   r2_score(data_y, preds), \
							   explained_variance_score(data_y, preds)])) 	