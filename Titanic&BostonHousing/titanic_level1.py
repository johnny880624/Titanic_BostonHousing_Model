"""
File: titanic_level1.py
Name: 
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python codes. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle. This model is the most flexible one among all
levels. You should do hyperparameter tuning and find the best model.
"""

import math
from util import *
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be processed
	:param data: dict[str: list], key is the column name, value is its data
	:param mode: str, indicating the mode we are using
	:param training_data: dict[str: list], key is the column name, value is its data
						  (You will only use this when mode == 'Test')
	:return data: dict[str: list], key is the column name, value is its data
	"""
	# index: 1:Survived 2:Pclass 5:Sex 6:Age 7:SibSp 8:Parch 10:Fare 12:Embarked
	key_dict = {}
	with open(filename, 'r') as f:
		first_line = True
		for line in f:
			line = line.replace('\n', '')
			data_lst = line.split(',')
			if first_line:
				if mode == 'Train':
					for i in range(len(data_lst)):
						if i == 1 or i == 2 or i == 4 or i == 5 or i == 6 or i == 7 or i == 9 or i == 11:
							key_dict[i] = data_lst[i]
							data[key_dict[i]] = []
							first_line = False
				else:
					for i in range(len(data_lst)):
						if i == 1 or i == 3 or i == 4 or i == 5 or i == 6 or i == 8 or i == 10:
							key_dict[i] = data_lst[i]
							data[key_dict[i]] = []
							first_line = False
			else:
				if mode == 'Train':
					# Age 是否有 missing data
					if len(data_lst[6]) == 0:
						pass
					elif len(data_lst[12]) == 0:
						pass
					else:
						for i in range(len(data_lst)):
							# Survived
							if i == 1:
								data['Survived'].append(int(data_lst[i]))
							# Pclass
							elif i == 2:
								data['Pclass'].append(int(data_lst[i]))
							# Sex
							elif i == 5:
								if data_lst[i] == 'male':
									data['Sex'].append(1)
								else:
									data['Sex'].append(0)
							# Age
							elif i == 6:
								data['Age'].append(float(data_lst[i]))
							# SibSp
							elif i == 7:
								data['SibSp'].append(int(data_lst[i]))
							# Parch
							elif i == 8:
								data['Parch'].append(int(data_lst[i]))
							# Fare
							elif i == 10:
								data['Fare'].append(float(data_lst[i]))
							# Embarked
							elif i == 12:
								if data_lst[i] == 'S':
									data['Embarked'].append(0)
								elif data_lst[i] == 'C':
									data['Embarked'].append(1)
								elif data_lst[i] == 'Q':
									data['Embarked'].append(2)
				else:
					for i in range(len(data_lst)):
						# Pclass
						if i == 1:
							data['Pclass'].append(int(data_lst[i]))
						# Sex
						elif i == 4:
							if data_lst[i] == 'male':
								data['Sex'].append(1)
							else:
								data['Sex'].append(0)
						# Age
						elif i == 5:
							if len(data_lst[5]) == 0:
								age_lst = training_data['Age']
								age_total = sum(age_lst)
								age_avg = age_total / len(training_data['Age'])
								age_avg = round(age_avg, 3)
								data['Age'].append(age_avg)
							else:
								data['Age'].append(float(data_lst[i]))
						# SibSp
						elif i == 6:
							data['SibSp'].append(int(data_lst[i]))
						# Parch
						elif i == 7:
							data['Parch'].append(int(data_lst[i]))
						# Fare
						elif i == 9:
							if len(data_lst[9]) == 0:
								fare_lst = training_data['Fare']
								fare_total = sum(fare_lst)
								fare_avg = fare_total / len(training_data['Fare'])
								fare_avg = round(fare_avg, 3)
								data['Fare'].append(fare_avg)
							else:
								data['Fare'].append(float(data_lst[i]))
						# Embarked
						elif i == 11:
							if data_lst[i] == 'S':
								data['Embarked'].append(0)
							elif data_lst[i] == 'C':
								data['Embarked'].append(1)
							elif data_lst[i] == 'Q':
								data['Embarked'].append(2)
	# print(data)
	return data


def one_hot_encoding(data: dict, feature: str):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: dict[str, list], remove the feature column and add its one-hot encoding features
	"""

	tem_lst_Pclass_0 = []
	tem_lst_Pclass_1 = []
	tem_lst_Pclass_2 = []
	tem_lst_Sex_0 = []
	tem_lst_Sex_1 = []
	tem_lst_Embarked_0 = []
	tem_lst_Embarked_1 = []
	tem_lst_Embarked_2 = []

	for i in range(len(data[feature])):
		if feature == 'Pclass':
			if data['Pclass'][i] == 1:
				tem_lst_Pclass_0.append(1)
			else:
				tem_lst_Pclass_0.append(0)
			if data['Pclass'][i] == 2:
				tem_lst_Pclass_1.append(1)
			else:
				tem_lst_Pclass_1.append(0)
			if data['Pclass'][i] == 3:
				tem_lst_Pclass_2.append(1)
			else:
				tem_lst_Pclass_2.append(0)

		elif feature == 'Sex':
			# famale
			if data['Sex'][i] == 0:
				tem_lst_Sex_0.append(1)
			else:
				tem_lst_Sex_0.append(0)
				# male
			if data['Sex'][i] == 1:
				tem_lst_Sex_1.append(1)
			else:
				tem_lst_Sex_1.append(0)

		elif feature == 'Embarked':
			if data['Embarked'][i] == 0:
				tem_lst_Embarked_0.append(1)
				tem_lst_Embarked_1.append(0)
				tem_lst_Embarked_2.append(0)
			elif data['Embarked'][i] == 1:
				tem_lst_Embarked_0.append(0)
				tem_lst_Embarked_1.append(1)
				tem_lst_Embarked_2.append(0)
			elif data['Embarked'][i] == 2:
				tem_lst_Embarked_0.append(0)
				tem_lst_Embarked_1.append(0)
				tem_lst_Embarked_2.append(1)

	if feature == 'Pclass':
		data['Pclass_0'] = tem_lst_Pclass_0
		data['Pclass_1'] = tem_lst_Pclass_1
		data['Pclass_2'] = tem_lst_Pclass_2
		data.pop('Pclass')
	elif feature == 'Sex':
		data['Sex_0'] = tem_lst_Sex_0
		data['Sex_1'] = tem_lst_Sex_1
		data.pop('Sex')
	elif feature == 'Embarked':
		data['Embarked_0'] = tem_lst_Embarked_0
		data['Embarked_1'] = tem_lst_Embarked_1
		data['Embarked_2'] = tem_lst_Embarked_2
		data.pop('Embarked')
	return data


def normalize(data: dict):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:return data: dict[str, list], key is the column name, value is its normalized data
	"""
	# 我自己的方法
	# for key, val in data.items():
	# 	lst = []
	# 	max_val = max(val)
	# 	min_val = min(val)
	# 	for i in range(len(val)):
	# 		# 分母不能為0
	# 		if min_val == max_val:
	# 			lst.append(0)
	# 		else:
	# 			x = (val[i] - min_val) / (max_val - min_val)
	# 			lst.append(x)
	# 	data[key] = lst

	# Karen's method 1
	# for key, val in data.items():
	# 	val is a list
	# 	lst = []
	# 	max_val = max(val)
	# 	min_val = min(val)
	# 	max_minus_min = max(val) - min(val)
	# 	for value in val:
	# 		if max_minus_min > 0:
	# 			value = (value - min(val))/max_minus_min
	# 			lst.append(value)
	# 	data[key] = lst

	for key, val in data.items():
		# val is a list
		max_val = max(val)
		min_val = min(val)
		max_minus_min = max(val) - min(val)
		for i in range(len(val)):
			if max_minus_min > 0:
				value = (val[i] - min(val))/max_minus_min


	# Karen's method2
	for x in data.keys():
		lst = []
		max_minus_min = max(data[x]) - min(data[x])
		# print(max_minus_min)
		for j in data[x]:
			# dict[x] is a lst
			# j is one of elements of lst
			if max_minus_min > 0:
				new_ele = (j - min(data[x])) / max_minus_min
				lst.append(new_ele)
		data[x] = lst



	return data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
	"""
	:param inputs: dict[str, list], key is the column name, value is its data
	:param labels: list[int], indicating the true label for each data
	:param degree: int, degree of polynomial features
	:param num_epochs: int, the number of epochs for training
	:param alpha: float, known as step size or learning rate
	:return weights: dict[str, float], feature name and its weight
	"""
	# Step 1 : Initialize weights
	weights = {}  # feature => weight

	keys = list(inputs.keys())
	if degree == 1:
		for i in range(len(keys)):
			weights[keys[i]] = 0
		print(weights)
	elif degree == 2:
		for i in range(len(keys)):
			weights[keys[i]] = 0
		for i in range(len(keys)):
			for j in range(i, len(keys)):
				weights[keys[i] + keys[j]] = 0
	# print(len(weights))

	if degree == 1:
		# start training
		for epochs in range(num_epochs):
			for i in range(len(labels)):
				feature_vector = {}
				label = labels[i]
				for key, val in inputs.items():
					feature_vector[key] = val[i]
				k = dotProduct(weights, feature_vector)
				h = sigmoid(k)
				update = round(-1 * alpha * (h - label), 3)
				increment(weights, update, feature_vector)

	elif degree == 2:
		for epochs in range(num_epochs):
			for i in range(len(labels)):
				label = labels[i]
				feature_vector = {}
				for key, val in inputs.items():
					feature_vector[key] = val[i]
				for x in range(len(keys)):
					for y in range(x, len(keys)):
						# feature_vector[keys[x] + keys[y]] = val[x] * val[y](錯的)
						feature_vector[keys[x] + keys[y]] = inputs[keys[x]][i] * inputs[keys[y]][i]
				k = dotProduct(weights, feature_vector)
				h = sigmoid(k)
				update = -1 * alpha * (h - label)
				increment(weights, update, feature_vector)

	return weights


def sigmoid(k):
	return 1/(1+math.exp(-k))

