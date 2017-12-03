import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split
import pprint
import seaborn as sns
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import tree
from sklearn import naive_bayes
from sklearn import svm 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from data_util import *


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


# Reads in data
data = pd.read_csv('./data/churn_data.csv')

#Remove Customer ID as it is not relevant to our data
del data['CustID']

#Assign Dummy Variables
data = pd.get_dummies(data, columns = cat_features(data))

#Remove 'Churn_no' - As '1' on 'ChurnYes' equals Yes, and '0' equals no. So there
# is no need for two columns of the same information
del data['Churn_No']

#Rename Churn yes to Churn. (1 = Yes, 0 = No)
data.rename(columns={'Churn_Yes':'Churn'}, inplace=True)

#Do the same for Gender. Will remove 'Gender_Female' so in the case a '0' in 
# 'Gender_male' would reprent a female while a '1' would represent a male
del data['Gender_Female']

#Rename Gender_Male to Gender. (1 = male, 0 = female)
data.rename(columns={'Gender_Male':'Gender'}, inplace=True)

#Delete low income, as high income already can represent all the data it holds
del data['Income_Lower']

#Rename income upper to income. 1 = upper income, 0 = lower income.
data.rename(columns={'Income_Upper':'Income'}, inplace=True)

#Assign x variables
features = list(data)

#Remove response variable from features
features.remove('Churn')

#Set response variable
response = data['Churn']

#Set x and y data
data_x = data[features]
data_y = data['Churn']

#Split training and test sets
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)


#---------Model Selection---------#

# 1
#RandomForestClassifier
# Build a sequence of models for different n_est and depth values. **NOTE: nest=10, depth=None is equivalent to the default.
n_est = [5, 10, 50, 100]
depth = [3, 6, None]
for n in n_est:
	for dp in depth:
		# Create model and fit.
		mod = ensemble.RandomForestClassifier(n_estimators=n, max_depth=dp)
		mod.fit(x_train, y_train)

		# Make predictions - both class labels and predicted probabilities.
		preds = mod.predict(x_test)
		#print('---------- EVALUATING MODEL: n_estimators = ' + str(n_est) + ', depth =' + str(dp) + ' -------------------')
		print('---------- EVALUATING MODEL: n_estimators = ' + str(n) + ', depth =' + str(dp) + ' -------------------')
		# Look at results.
		print_multiclass_classif_error_report(y_test, preds)

#2
#Decision Tree
# Split training and test sets from main set.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

# Build and evaluate 2 models: one with Gini Impurity criteria and one with Information Gain criteria.
print('----------- DTREE WITH GINI IMPURITY CRITERION ------------------')
dtree_gini_mod = tree.DecisionTreeClassifier(criterion='gini')
dtree_gini_mod.fit(x_train, y_train)
preds_gini = dtree_gini_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds_gini)

print('\n----------- DTREE WITH ENTROPY CRITERION -----------------------')
dtree_entropy_mod = tree.DecisionTreeClassifier(criterion='entropy')
dtree_entropy_mod.fit(x_train, y_train)
preds_entropy = dtree_entropy_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds_entropy)

#3
#Niave Bayesian Classifier
# Split training and test sets from main set.

# Convert the different class labels to unique numbers with label encoding.
le = preprocessing.LabelEncoder()
data_y = le.fit_transform(data_y)

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

# Build and evaluate the model
gnb_mod = naive_bayes.GaussianNB()
gnb_mod.fit(x_train, y_train)
preds = gnb_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds)

## Illustrate recoding numeric classes back into original (text-based) labels.
#y_test_labs = le.inverse_transform(y_test)
#pred_labs = le.inverse_transform(preds)
#print('(Actual, Predicted): \n' + str(zip(y_test_labs, pred_labs)))

#4 
#SVM
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)
# Build a sequence of models for different n_est and depth values. **NOTE: c=1.0 is equivalent to the default.
cs = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
for c in cs:
	# Create model and fit.
	mod = svm.SVC(C=c)
	mod.fit(x_train, y_train)

	# Make predictions - both class labels and predicted probabilities.
	preds = mod.predict(x_test)
	print('---------- EVALUATING MODEL: C = ' + str(c) + ' -------------------')
	# Look at results.
	print_multiclass_classif_error_report(y_test, preds)
	
#5 
#K-nearest neighbor

#Normalize predictors
#data_x = preprocessing.normalize(data_x, axis = 0)
#Error: float() argument must be a string or number

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)


# Build a sequence of models for k = 2, 4, 6, 8, ..., 20.
ks = [2, 3, 6, 8, 10, 12, 14, 16, 18, 20]
for k in ks:
	# Create model and fit.
	mod = neighbors.KNeighborsClassifier(n_neighbors=k)
	mod.fit(x_train, y_train)

	# Make predictions - both class labels and predicted probabilities.
	preds = mod.predict(x_test)
	print('---------- EVALUATING MODEL: k = ' + str(k) + ' -------------------')
	# Look at results.
	print('Accuracy: ' + str(accuracy_score(y_test, preds)))
	print('Precison: ' + str(precision_score(y_test, preds)))
	print('Recall: ' + str(recall_score(y_test, preds)))
	print('F1: ' + str(f1_score(y_test, preds)))
	print('ROC AUC: ' + str(roc_auc_score(y_test, preds)))
	print("Confusion Matrix:\n" + str(confusion_matrix(y_test, preds)))

# #6
# #Tuning up Random Forest Classification

# # Create training and test sets for later use.
# x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

# # Optimize a RF classifier and test with grid search.
# # Below - notice that n_estimators and max_depth are both params of RF. 
# param_grid = {'n_estimators':[5, 10, 50, 100], 'max_depth':[3, 6, None]} 

# # Find the best RF and use that. Do a 5-fold CV and score with f1 macro.
# optimized_rf = GridSearchCV(ensemble.RandomForestClassifier(), param_grid, cv=5, scoring='f1_macro')

# # Fit the optimized RF just like it is a single model
# optimized_rf.fit(x_train, y_train)

# #Returns score of .846
# print('Grid Search Test Score (Random Forest): ' + str(optimized_rf.score(x_test, y_test)))

# # Here is a Bagging classifier that builds many SVM's.
# bagging_mod = ensemble.BaggingClassifier(linear_model.LogisticRegression(), n_estimators=200)
# k_fold = KFold(n_splits=5, shuffle=True, random_state=4) 
# bagging_mod_scores = cross_val_score(bagging_mod, data_x, data_y, scoring='f1_macro', cv=k_fold)
# print('CV Scores (Bagging NB) ' + str(bagging_mod_scores))

# # Here is a basic voting classifier with CV and Grid Search.
# m1 = svm.SVC()
# m2 = ensemble.RandomForestClassifier()
# m3 = naive_bayes.GaussianNB()
# voting_mod = ensemble.VotingClassifier(estimators=[('svm', m1), ('rf', m2), ('nb', m3)], voting='hard')

# # Set up params for combined Grid Search on the voting model. Notice the convention for specifying 
# # parameters foreach of the different models.
# param_grid = {'svm__C':[0.2, 0.5, 1.0, 2.0, 5.0, 10.0], 'rf__n_estimators':[5, 10, 50, 100], 'rf__max_depth': [3, 6, None]}
# best_voting_mod = GridSearchCV(estimator=voting_mod, param_grid=param_grid, cv=5)
# best_voting_mod.fit(x_train, y_train)
# print('Voting Ensemble Model Test Score: ' + str(best_voting_mod.score(x_test, y_test)))
# #Output: Voting Enselble Model Test Score: .820

#------------------------------------------Test model on validation data------------------------------------------#
#Read in csv
data_v = pd.read_csv('./data/churn_validation.csv')

#Remove Customer ID as it is not relevant to our data_v
del data_v['CustID']

#Assign Dummy Variables
data_v = pd.get_dummies(data_v, columns = cat_features(data_v))

#Remove 'Churn_no' - As '1' on 'ChurnYes' equals Yes, and '0' equals no. So there
# is no need for two columns of the same information
del data_v['Churn_No']

#Rename Churn yes to Churn. (1 = Yes, 0 = No)
data_v.rename(columns={'Churn_Yes':'Churn'}, inplace=True)

#Do the same for Gender. Will remove 'Gender_Female' so in the case a '0' in 
# 'Gender_male' would reprent a female while a '1' would represent a male
del data_v['Gender_Female']

#Rename Gender_Male to Gender. (1 = male, 0 = female)
data_v.rename(columns={'Gender_Male':'Gender'}, inplace=True)

#Delete low income, as high income already can represent all the data_v it holds
del data_v['Income_Lower']

#Rename income upper to income. 1 = upper income, 0 = lower income.
data_v.rename(columns={'Income_Upper':'Income'}, inplace=True)

#Assign x variables
features = list(data_v)

#Remove response variable from features
features.remove('Churn')

#Set response variable
response = data_v['Churn']

#Set x and y data_v
data_v_x = data_v[features]
data_v_y = data_v['Churn']

#Split training and test sets
x_train, x_test, y_train, y_test = train_test_split(data_v_x, data_v_y, test_size = 0.3, random_state = 4)

#RandomForestClassifier
# Build a sequence of models for different n_est and depth values. **NOTE: nest=10, depth=None is equivalent to the default.
n_est = [5, 10, 50, 100]
depth = [3, 6, None]
for n in n_est:
	for dp in depth:
		# Create model and fit.
		mod = ensemble.RandomForestClassifier(n_estimators=n, max_depth=dp)
		mod.fit(x_train, y_train)

		# Make predictions - both class labels and predicted probabilities.
		preds = mod.predict(x_test)
		#print('---------- EVALUATING MODEL: n_estimators = ' + str(n_est) + ', depth =' + str(dp) + ' -------------------')
		print('---------- EVALUATING MODEL: n_estimators = ' + str(n) + ', depth =' + str(dp) + ' -------------------')
		# Look at results.
		print_multiclass_classif_error_report(y_test, preds)
