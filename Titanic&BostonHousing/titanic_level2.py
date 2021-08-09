"""
File: titanic_level2.py
Name: 
----------------------------------
This file builds a machine learning algorithm by pandas and sklearn libraries.
We'll be using pandas to read in dataset, store data into a DataFrame,
standardize the data by sklearn, and finally train the model and
test it on kaggle. Hyperparameters are hidden by the library!
This abstraction makes it easy to use but less flexible.
You should find a good model that surpasses 77% test accuracy on kaggle.
"""

import math
import pandas as pd
from sklearn import preprocessing, linear_model
from sklearn.preprocessing import PolynomialFeatures
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'

nan_cache = {}


def data_preprocess(filename, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
						  (You will only use this when mode == 'Test')
	:return: Tuple(data, labels), if the mode is 'Train'
			 data, if the mode is 'Test'0
	"""
	data = pd.read_csv(filename)
	labels = None
	if mode == 'Train':
		data = data.drop(columns=['PassengerId','Name','Ticket','Cabin'])
		data = data.dropna()
		labels = data.pop('Survived')
		# 把training feature 的平均數放在一個箱子存起來
		nan_cache['Age'] = round(data['Age'].mean(), 3)
		nan_cache['Fare'] = round(data['Fare'].mean(), 3)

	elif mode == 'Test':
		data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
		data['Age'] = data['Age'].fillna(nan_cache['Age'])
		data['Fare'] = data['Fare'].fillna(nan_cache['Fare'])
		# print(data.count())
	data.loc[data.Sex == 'male', 'Sex'] = 1
	data.loc[data.Sex == 'female', 'Sex'] = 0

	data.loc[data.Embarked == 'S', 'Embarked'] = 0
	data.loc[data.Embarked == 'C', 'Embarked'] = 1
	data.loc[data.Embarked == 'Q', 'Embarked'] = 2

	if mode == 'Train':
		return data, labels
	elif mode == 'Test':
		return data


def one_hot_encoding(data, feature):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: DataFrame, remove the feature column and add its one-hot encoding features
	"""
	# 內建的 One_hot_Encoding
	data = pd.get_dummies(data, columns=[feature])
	# if feature == 'Sex':
	# 	# famale
	# 	data['Sex_0'] = 0
	# 	data.loc[data.Sex == 0, 'Sex_0'] = 1
	# 	# male
	# 	data['Sex_1'] = 0
	# 	data.loc[data.Sex == 1, 'Sex_1'] = 1
	# 	data.pop('Sex')
	#
	# elif feature == 'Pclass':
	# 	# ‘Pclass_0
	# 	data['Pclass_0'] = 0
	# 	data.loc[data.Pclass == 1, 'Pclass_0'] = 1
	# 	# 'Pclass_1'
	# 	data['Pclass_1'] = 0
	# 	data.loc[data.Pclass == 2, 'Pclass_1'] = 1
	# 	# 'Pclass_2'
	# 	data['Pclass_2'] = 0
	# 	data.loc[data.Pclass == 3, 'Pclass_2'] = 1
	# 	data.pop('Pclass')
	#
	# elif feature == 'Embarked':
	# 	# Embarked_0
	# 	data['Embarked_0'] = 0
	# 	data.loc[data.Embarked == 0, 'Embarked_0'] = 1
	# 	# Embarked_1
	# 	data['Embarked_1'] = 0
	# 	data.loc[data.Embarked == 1, 'Embarked_1'] = 1
	# 	# Embarked_2
	# 	data['Embarked_2'] = 0
	# 	data.loc[data.Embarked == 2, 'Embarked_2'] = 1
	# 	data.pop('Embarked')

	return data


def standardization(data, mode='Train'):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param mode: str, indicating the mode we are using (either Train or Test)
	:return data: DataFrame, standardized features
	"""

	standardizer = preprocessing.StandardScaler()
	if mode == 'Train':
		data = standardizer.fit_transform(data)
	elif mode == 'Test':
		data = standardizer.transform(data)
	return data


def main():
	"""
	You should call data_preprocess(), one_hot_encoding(), and
	standardization() on your training data. You should see ~80% accuracy
	on degree1; ~83% on degree2; ~87% on degree3.
	Please write down the accuracy for degree1, 2, and 3 respectively below
	(rounding accuracies to 8 decimals)
	TODO: real accuracy on degree1 -> ______________________
	TODO: real accuracy on degree2 -> ______________________
	TODO: real accuracy on degree3 -> ______________________
	"""
	train_data_tuple = data_preprocess(TRAIN_FILE, mode='Train')
	test_data = data_preprocess(TEST_FILE, mode='Test')

	train_data = train_data_tuple[0]
	print(train_data)
	label = train_data_tuple[1]

	train_data = one_hot_encoding(train_data, 'Sex')
	train_data = one_hot_encoding(train_data, 'Pclass')
	train_data = one_hot_encoding(train_data, 'Embarked')

	# degree 1 without standardization
	h = linear_model.LogisticRegression(max_iter=10000)
	standardizer = preprocessing.StandardScaler()
	train_data = standardizer.fit_transform(train_data)
	classifier = h.fit(train_data, label)
	acc = classifier.score(train_data, label)
	print('real accuracy on degree1:', acc)

	# degree 2 without standardization
	poly_feature_extractor = PolynomialFeatures(degree=2)
	poly_data = poly_feature_extractor.fit_transform(train_data)
	classifier_poly = h.fit(poly_data, label)
	acc2 = classifier_poly.score(poly_data, label)
	print('real accuracy on degree2:', acc2)

	# degree 3 without standardization
	poly_feature_extractor_degree3 = PolynomialFeatures(degree=3)
	poly_data_3 = poly_feature_extractor_degree3.fit_transform(train_data)
	classifier_poly_3 = h.fit(poly_data_3, label)
	acc3 = classifier_poly_3.score(poly_data_3, label)
	print('real accuracy on degree3:', acc3)




if __name__ == '__main__':
	main()
