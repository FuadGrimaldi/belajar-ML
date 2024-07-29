import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing, datasets, tree
from sklearn.model_selection import train_test_split, cross_val_score

data = [[12000000, 33], [35000000, 45], [40000000, 23], [65000000, 26], [9000000, 29]]
# _________________________________Normalization____________________________________
scaler = MinMaxScaler()
scaler.fit(data)
print(scaler.transform(data))

# _________________________________Standardization____________________________________
scaler = preprocessing.StandardScaler().fit(data)
data = scaler.transform(data)
print(data)

# __________________________Training Set dan Test Set________________________________
# X_data = range(10)
# y_data = range(10)
 
# print("random_state ditentukan")
# for i in range(3):
#     X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state = 42)
#     print(y_test)
 
 
# print("random_state tidak ditentukan")
# for i in range(3):
#     X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state = None)
#     print(y_test)


# # __________________________Training Set dan Test Set________________________________
# iris = datasets.load_iris()
# x = iris.data
# y = iris.target
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
# print(len(x_test))


# # _______________________SKLearn Cross Validation Split________________________________
# iris = datasets.load_iris()
# x = iris.data
# y = iris.target
# clf = tree.DecisionTreeClassifier()
# scores = cross_val_score(clf, x, y, cv=5)
# print(scores)


# df = pd.DataFrame(data, columns=['gaji', 'umur'])
# print(df)