"""
File: boston_housing_competition.py
Name: 
--------------------------------
This file demonstrates how to analyze boston
housing dataset. Students will upload their 
results to kaggle.com and compete with people
in class!

You are allowed to use pandas, sklearn, or build the
model from scratch! Go data scientist!
"""
import pandas as pd
from sklearn import linear_model, preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn import decomposition

TRAIN_DATA = 'boston_housing/train.csv'
TEST_DATA = 'boston_housing/test.csv'

def main():
	train_data_tuple = data_preprocessing(TRAIN_DATA, mode='Train')


	test_data = data_preprocessing(TEST_DATA, mode='Test')

	train_data = train_data_tuple[0]
	labels = train_data_tuple[1]
	# Extract features
	train_data = train_data[['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'tax', 'ptratio', 'lstat']]

	# standardization
	standardizer = preprocessing.StandardScaler()
	train_data = standardizer.fit_transform(train_data)

	# Principal Component Analysis
	pca = decomposition.PCA(n_components=9)
	train_data = pca.fit_transform(train_data)

	# Degree 2 Polynomial
	poly_feature_extractor = PolynomialFeatures(degree=3)
	train_data = poly_feature_extractor.fit_transform(train_data)

	# Training
	h = linear_model.LinearRegression()
	classifier = h.fit(train_data, labels)
	acc = classifier.score(train_data, labels)
	print(acc)

	# Test
	test_data_org = test_data
	# test_data = test_data[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']]
	test_data = test_data[['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'tax', 'ptratio', 'lstat']]
	test_data = standardizer.transform(test_data)
	test_data = pca.transform(test_data)
	test_data = poly_feature_extractor.transform(test_data)
	predictions = classifier.predict(test_data)
	print(predictions)
	# print(len(predictions))
	# test(predictions, 'submission11.csv', test_data_org)

def data_preprocessing(filename, mode ='Train', training_data =None):
	data = pd.read_csv(filename)
	labels = None
	if mode == 'Train':
		# train_data = data.corr(method='pearson')
		# pd.set_option('display.max_rows', None)
		# pd.set_option('display.max_columns', None)
		# pd.set_option('display.width', None)
		# pd.set_option('display.max_colwidth', -1)
		# print(train_data)
		# data = data.drop(columns='ID')
		train_data_show = data.corr(method='spearman')
		print(train_data_show)
		labels = data.pop('medv')
	# else:
	# 	data = data.drop(columns='ID')

	if mode == 'Train':
		return data, labels
	else:
		return data


def standardization(data, mode ='Train'):
	standardizer = preprocessing.StandardScaler()
	if mode == 'Train':
		data = standardizer.fit_transform(data)
	else:
		data = standardizer.fit_transform(data)
	return data


def test(predictions, filename, test_data):
	"""
	: param predictions: numpy.array, a list-like data structure that stores 0's and 1's
	: param filename: str, the filename you would like to write the results to
	"""
	print('\n==========================')
	print('Writing predictions to ...')
	print(filename)
	id_lst2 = []
	id_lst = test_data.ID
	for i in range(len(id_lst)):
		id_lst2.append(id_lst[i])
	print(id_lst2)
	print(len(id_lst2))
	with open(filename, 'w') as out:
		out.write('ID,medv\n')
		count = 0
		for ans in predictions:
			out.write(str(id_lst2[count]) + ','+str(ans)+'\n')
			count += 1


	print('\n==========================')


if __name__ == '__main__':
	main()
