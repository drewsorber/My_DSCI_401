# cd C:\GitHub\my_dsci_401
import pandas as pd
import matplotlib.pyplot as plt
import pprint
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import svm 
from sklearn import naive_bayes
from sklearn import ensemble 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
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

#Read in data
df = pd.read_csv('./2015-2016datatest.csv')

#Delete Team and playername variablea, not relevant 
del df['TEAM']
del df['Player']

#set features and response variable
features = list(df)
features.remove('SALARY')
data_x = df[features]
data_y = df['SALARY']

#Impute missing values 
#Upon looking at the data it became apparent that most players with missing data were players that have
#Very low play time. This means that using the mean or median would put them on par with more average players in the league
#even though they are clealy below average, as they do not recieve a lot of playing time. I chose most frequent as there
#Are a lot of players with low play time, and in turn low statistics that are below average. Using most frequent will
#impute values that put them closer to the lower level players in the league.
imp = preprocessing.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
data_x = imp.fit_transform(data_x)

#loops through scatter matrices 10 variables at time as suggested in class.
#Commented out as this takes 'forever' for my laptop to load
# j=0
# for i in range(0,len(features)/5):
	# sm = pd.plotting.scatter_matrix(df[features[j:j+5] + ['SALARY']])
	# #plt.tight_layout()
	# plt.show()
	# j+=5

	

	
#-------------------------------------Model Building-------------------------------------#


#1. Basic Linear Regression

# Create a least squares linear regression model.
model = linear_model.LinearRegression()

# Training data used to train the algorithm, test is to assess performance.
x_train, x_test, y_train, y_test = data_x, data_x, data_y, data_y

# Fit the model.
model.fit(x_train,y_train)

# Make predictions on test data and look at the results.
preds = model.predict(x_test)
pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))


#2. Linear Regression with proper train/test split
# Create a least squares linear regression model.
model = linear_model.LinearRegression()

# Split training and test sets from main set. Note: random_state just enables results to be repeated.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)

# Fit the model.
model.fit(x_train,y_train)

# Make predictions on test data and look at the results.
preds = model.predict(x_test)
pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))
print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), \
							   median_absolute_error(y_test, preds), \
							   r2_score(y_test, preds), \
							   explained_variance_score(y_test, preds)])) 





#3. Lasso Regression 

# Split training and test sets from main set. 
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)

# Fit the model.
# Create a least squares linear regression model.
base_mod = linear_model.LinearRegression()
base_mod.fit(x_train,y_train)

# Make predictions on test data and look at the results.
preds = base_mod.predict(x_test)
print('R^2 (Base Model): ' + str(r2_score(y_test, preds)))

# Show Lasso regression fits for different alphas.
alphas = [0.01, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
for a in alphas:
	 # Normalizing transforms all variables to number of standard deviations away from mean.
	 lasso_mod = linear_model.Lasso(alpha=a, normalize=True, fit_intercept=True)
	 lasso_mod.fit(x_train, y_train)
	 preds = lasso_mod.predict(x_test)
	 print('R^2 (Lasso Model with alpha=' + str(a) + '): ' + str(r2_score(y_test, preds)))
	 
