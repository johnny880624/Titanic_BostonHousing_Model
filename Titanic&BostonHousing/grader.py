#!/usr/bin/python3

import graderUtil
import util
import time
from collections import defaultdict
from util import *

grader = graderUtil.Grader()
titanic_level1 = grader.load('titanic_level1')
titanic_level2 = grader.load('titanic_level2')


############################################################
# Milestone 1: data preprocessing
############################################################

### 1a

# Basic sanity check for data preprocessing
def test1_0():
    ans = 712
    grader.require_is_equal(ans, len(titanic_level1.data_preprocess('titanic_data/train.csv', {})['Age']))
grader.add_basic_part('test1_0', test1_0, max_points=4, max_seconds=1, description="Milestone 1 - Number of data == 712")


def test1_1():
    ans = [0, 1, 1, 54.0, 0, 0, 51.8625, 0]
    data = titanic_level1.data_preprocess('titanic_data/train.csv', {})
    count = 5
    test = []
    for key, value in data.items():
        if key == 'Survived':
            test.append(value[count])
        elif key == 'Pclass':
            test.append(value[count])
        elif key == 'Sex':
            test.append(value[count])
        elif key == 'Age':
            test.append(value[count])
        elif key == 'SibSp':
            test.append(value[count])
        elif key == 'Parch':
            test.append(value[count])
        elif key == 'Fare':
            test.append(value[count])
        elif key == 'Embarked':
            test.append(value[count])
    grader.require_is_equal(ans, test)
grader.add_basic_part('test1_1', test1_1, max_points=4, max_seconds=1, description="Milestone 1 - 6th data")


def test1_2():
    ans = [0, 1, 1, 45.0, 1, 0, 83.475, 0]
    data = titanic_level1.data_preprocess('titanic_data/train.csv', {})
    count = 46
    test = []
    for key, value in data.items():
        if key == 'Survived':
            test.append(value[count])
        elif key == 'Pclass':
            test.append(value[count])
        elif key == 'Sex':
            test.append(value[count])
        elif key == 'Age':
            test.append(value[count])
        elif key == 'SibSp':
            test.append(value[count])
        elif key == 'Parch':
            test.append(value[count])
        elif key == 'Fare':
            test.append(value[count])
        elif key == 'Embarked':
            test.append(value[count])
    grader.require_is_equal(ans, test)
grader.add_basic_part('test1_2', test1_2, max_points=4, max_seconds=1, description="Milestone 1 - 47th data")


def test1_3():
    ans = 0
    train_data = titanic_level1.data_preprocess('titanic_data/train.csv', {})
    test = 0
    ignore_lst = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    for item in ignore_lst:
        if item in train_data:
            test += 1
    grader.require_is_equal(ans, test)
grader.add_basic_part('test1_3', test1_3, max_points=3, max_seconds=1, description="Milestone 1 - If PassengerId, Name, Ticket, Cabin are ignored")


def test2_0():
    ans = 418
    train_data = titanic_level1.data_preprocess('titanic_data/train.csv', {})
    grader.require_is_equal(ans, len(titanic_level1.data_preprocess('titanic_data/test.csv', {}, mode='Test', training_data=train_data)['Age']))
grader.add_basic_part('test2_0', test2_0, max_points=5, max_seconds=1, description="Milestone 2 - Number of data == 418")


def test2_1():
    ans = [3, 1, 29.642, 0, 0, 7.8958, 0]
    train_data = titanic_level1.data_preprocess('titanic_data/train.csv', {})
    data = titanic_level1.data_preprocess('titanic_data/test.csv', {}, mode='Test', training_data=train_data)
    count = 10
    test = []
    for key, value in data.items():
        if key == 'Survived':
            test.append(value[count])
        elif key == 'Pclass':
            test.append(value[count])
        elif key == 'Sex':
            test.append(value[count])
        elif key == 'Age':
            test.append(value[count])
        elif key == 'SibSp':
            test.append(value[count])
        elif key == 'Parch':
            test.append(value[count])
        elif key == 'Fare':
            test.append(value[count])
        elif key == 'Embarked':
            test.append(value[count])
    grader.require_is_equal(ans, test)
grader.add_basic_part('test2_1', test2_1, max_points=5, max_seconds=1, description="Milestone 2 - 11th data")


def test2_2():
    ans = [3, 1, 60.5, 0, 0, 34.567, 0]
    train_data = titanic_level1.data_preprocess('titanic_data/train.csv', {})
    data = titanic_level1.data_preprocess('titanic_data/test.csv', {}, mode='Test', training_data=train_data)
    count = 152
    test = []
    for key, value in data.items():
        if key == 'Survived':
            test.append(value[count])
        elif key == 'Pclass':
            test.append(value[count])
        elif key == 'Sex':
            test.append(value[count])
        elif key == 'Age':
            test.append(value[count])
        elif key == 'SibSp':
            test.append(value[count])
        elif key == 'Parch':
            test.append(value[count])
        elif key == 'Fare':
            test.append(value[count])
        elif key == 'Embarked':
            test.append(value[count])
    grader.require_is_equal(ans, test)
grader.add_basic_part('test2_2', test2_2, max_points=5, max_seconds=1, description="Milestone 2 - 153th data")


def test3_1():
    ans = [0, 1, 1, 1, 0, 0, 0, 1]
    train_data = titanic_level1.data_preprocess('titanic_data/train.csv', {})
    new_data = titanic_level1.one_hot_encoding(train_data, 'Sex')
    test = []
    count = 0
    for value in new_data['Sex_0']:
        test.append(value)
        if count == 7:
            break
        count += 1
    grader.require_is_equal(ans, test)
grader.add_basic_part('test3_1', test3_1, max_points=4, max_seconds=1, description="Milestone 3 - Sex_0 - Originally Female")

def test3_2():
    ans = [0, 1, 0, 1, 0, 1, 0, 0, 0]
    train_data = titanic_level1.data_preprocess('titanic_data/train.csv', {})
    new_data = titanic_level1.one_hot_encoding(train_data, 'Pclass')
    test = []
    count = 0
    for value in new_data['Pclass_0']:
        test.append(value)
        if count == 8:
            break
        count += 1
    grader.require_is_equal(ans, test)
grader.add_basic_part('test3_2', test3_2, max_points=4, max_seconds=1, description="Milestone 3 - Pclass_0 - Originally 1")

def test3_3():
    ans = [0, 0, 0, 0, 0, 0, 0, 0, 1]
    train_data = titanic_level1.data_preprocess('titanic_data/train.csv', {})
    new_data = titanic_level1.one_hot_encoding(train_data, 'Pclass')
    test = []
    count = 0
    for value in new_data['Pclass_1']:
        test.append(value)
        if count == 8:
            break
        count += 1
    grader.require_is_equal(ans, test)
grader.add_basic_part('test3_3', test3_3, max_points=4, max_seconds=1, description="Milestone 3 - Pclass_1 - Originally 2")

def test3_4():
    ans = False
    train_data = titanic_level1.data_preprocess('titanic_data/train.csv', {})
    new_data = titanic_level1.one_hot_encoding(train_data, 'Pclass')
    test = True if 'Pclass' in new_data else False
    grader.require_is_equal(ans, test)
grader.add_basic_part('test3_4', test3_4, max_points=3, max_seconds=1, description="Milestone 3 - remove the previous feature")


def test4_0():
    ans = 48.039
    train_data = titanic_level1.data_preprocess('titanic_data/train.csv', {})
    test = sum(titanic_level1.normalize(train_data)['Fare'])
    grader.require_is_equal(ans, round(test, 3))
grader.add_basic_part('test4_0', test4_0, max_points=5, max_seconds=1, description="Milestone 4 - Normalized Fare")


def test5_0():
    ans = 0.8104
    train_data = titanic_level1.data_preprocess('titanic_data/train.csv', {})
    train_data = titanic_level1.one_hot_encoding(train_data, 'Sex')
    train_data = titanic_level1.one_hot_encoding(train_data, 'Pclass')
    train_data = titanic_level1.one_hot_encoding(train_data, 'Embarked')
    labels = train_data.pop('Survived')
    labels = list(int(labels[i]) for i in range(len(labels)))
    train_data = titanic_level1.normalize(train_data)
    weights = titanic_level1.learnPredictor(train_data, labels, 1, 100, 0.1)
    test = round(evaluatePredictor(train_data, labels, weights, 2), 4)
    grader.require_is_equal(ans, test)
grader.add_basic_part('test5_0', test5_0, max_points=10, max_seconds=10, description="Milestone 5 - Classification Model degree 1")



# def test5_1():
#     ans = 0.8258
#     train_data = titanic_level1.data_preprocess('titanic_data/train.csv', {})
#     train_data = titanic_level1.one_hot_encoding(train_data, 'Sex')
#     train_data = titanic_level1.one_hot_encoding(train_data, 'Pclass')
#     train_data = titanic_level1.one_hot_encoding(train_data, 'Embarked')
#     labels = train_data.pop('Survived')
#     labels = list(int(labels[i]) for i in range(len(labels)))
#     train_data = titanic_level1.normalize(train_data)
#     weights = titanic_level1.learnPredictor(train_data, labels, 2, 100, 0.1)
#     test = round(evaluatePredictor(train_data, labels, weights, 2), 4)
#     grader.require_is_equal(ans, test)
# grader.add_basic_part('test5_1', test5_1, max_points=11, max_seconds=10, description="Milestone 5 - Classification Model degree 2")


def test6_0():
    ans = {'Pclass': 712, 'Sex': 712, 'Age': 712, 'SibSp': 712, 'Parch': 712, 'Fare': 712, 'Embarked': 712}
    train_data, _ = titanic_level2.data_preprocess('titanic_data/train.csv')
    grader.require_is_equal(ans, dict(train_data.count()))
grader.add_basic_part('test6_0', test6_0, max_points=3, max_seconds=1, description="Milestone 6 - Training Data")


def test6_1():
    ans = [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    _, labels = titanic_level2.data_preprocess('titanic_data/train.csv')
    grader.require_is_equal(ans, list(labels)[0:13])
grader.add_basic_part('test6_1', test6_1, max_points=3, max_seconds=1, description="Milestone 6 - Training Data")


def test7_0():
    ans = {'Pclass': 418, 'Sex': 418, 'Age': 418, 'SibSp': 418, 'Parch': 418, 'Fare': 418, 'Embarked': 418}
    train_data, _ = titanic_level2.data_preprocess('titanic_data/train.csv')
    test_data = titanic_level2.data_preprocess('titanic_data/test.csv', mode="Test", training_data = train_data)
    grader.require_is_equal(ans, dict(test_data.count()))
grader.add_basic_part('test7_0', test7_0, max_points=3, max_seconds=1, description="Milestone 7 - Test Data")


def test7_1():
    ans = {'Pclass': 3, 'Sex': 1, 'Age': 29.642, 'SibSp': 0, 'Parch': 0, 'Fare': 7.8958, 'Embarked': 0}
    train_data, _ = titanic_level2.data_preprocess('titanic_data/train.csv')
    test_data = titanic_level2.data_preprocess('titanic_data/test.csv', mode="Test", training_data = train_data)
    grader.require_is_equal(ans, dict(test_data.iloc[10]))
grader.add_basic_part('test7_1', test7_1, max_points=3, max_seconds=1, description="Milestone 7 - 10th Test Data")


def test7_2():
    ans = {'Pclass': 3, 'Sex': 1, 'Age': 60.5, 'SibSp': 0, 'Parch': 0, 'Fare': 34.567, 'Embarked': 0}
    train_data, _ = titanic_level2.data_preprocess('titanic_data/train.csv')
    test_data = titanic_level2.data_preprocess('titanic_data/test.csv', mode="Test", training_data = train_data)
    grader.require_is_equal(ans, dict(test_data.iloc[152]))
grader.add_basic_part('test7_2', test7_2, max_points=3, max_seconds=1, description="Milestone 7 - 153th Test Data")


def test8_1():
    ans = [0, 1, 1, 1, 0, 0, 0, 1]
    train_data, _ = titanic_level2.data_preprocess('titanic_data/train.csv')
    print(train_data)
    new_data = titanic_level2.one_hot_encoding(train_data, 'Sex')
    print(new_data)
    test = []
    count = 0
    for value in new_data['Sex_0']:
        test.append(value)
        if count == 7:
            break
        count += 1
    grader.require_is_equal(ans, test)
grader.add_basic_part('test8_1', test8_1, max_points=3, max_seconds=1, description="Milestone 8 - Sex_0 - Originally Female")


def test8_2():
    ans = [0, 1, 0, 1, 0, 1, 0, 0, 0]
    train_data, _ = titanic_level2.data_preprocess('titanic_data/train.csv')
    new_data = titanic_level2.one_hot_encoding(train_data, 'Pclass')
    test = []
    count = 0
    for value in new_data['Pclass_0']:
        test.append(value)
        if count == 8:
            break
        count += 1
    grader.require_is_equal(ans, test)
grader.add_basic_part('test8_2', test8_2, max_points=3, max_seconds=1, description="Milestone 8 - Pclass_0 - Originally 1")


def test8_3():
    ans = [0, 0, 0, 0, 0, 0, 0, 0, 1]
    train_data, _ = titanic_level2.data_preprocess('titanic_data/train.csv')
    new_data = titanic_level2.one_hot_encoding(train_data, 'Pclass')
    test = []
    count = 0
    for value in new_data['Pclass_1']:
        test.append(value)
        if count == 8:
            break
        count += 1
    grader.require_is_equal(ans, test)
grader.add_basic_part('test8_3', test8_3, max_points=3, max_seconds=1, description="Milestone 8 - Pclass_1 - Originally 2")


def test9_0():
    ans = 0
    train_data, _ = titanic_level2.data_preprocess('titanic_data/train.csv')
    test = sum(titanic_level2.standardization(train_data))
    grader.require_is_equal(ans, round(test[5], 3))
grader.add_basic_part('test9_0', test9_0, max_points=5, max_seconds=1, description="Milestone 9 - Standardized Fare")


def evaluatePredictor(inputs, labels, weights, degree):
    """
    :param inputs: dict[str, list], key is the column name, value is its data
    :param labels: list[int], indicating the true label for each data
    :param weights: dict[str, float], feature name and its weight
    :param degree: int, degree of polynomial features
    """
    prediction_list = []
    keys = list(inputs.keys())
    for i in range(len(labels)):
        feature = {}
        if degree == 1:
            for j in range(len(keys)):
                feature[keys[j]] = inputs[keys[j]][i]
        else:
            for j in range(len(keys)):
                feature[keys[j]] = inputs[keys[j]][i]
            for j in range(len(keys)):
                for k in range(j, len(keys)):
                    feature[keys[j] + keys[k]] = inputs[keys[j]][i] * inputs[keys[k]][i]

        prediction = dotProduct(weights, feature)
        prediction = 1 if prediction > 0 else 0
        prediction_list.append(prediction)
    accuracy = 0
    for i in range(len(labels)):
        if prediction_list[i] == labels[i]:
            accuracy += 1
    accuracy = accuracy / len(labels)
    if accuracy > 0.8:
        print('Congratulations! Your accuracy is over 80 percent, it is %s percent now.' % (round(accuracy * 100, 2)))
    else:
        print('Oops...You need to check your model again. The accuracy is %s percent now.' % (round(accuracy * 100, 2)))
    return accuracy

grader.grade()
