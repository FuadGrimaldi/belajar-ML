import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
data_path = 'DataSet/iris.csv'
df_iris = pd.read_csv(data_path)
df_iris.drop(['Id'], axis=1)
# memisahkan atribut dan label
x = df_iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df_iris['Species']
# membagi dataset menjadi data latih & data uji (data training & data test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)
# membuat model decision tree
tree_model = DecisionTreeClassifier()
# melatih model dengan menggunakan data latih
tree_model = tree_model.fit(x_train, y_train)
# evaluasi model
y_pred = tree_model.predict(x_test)
acc_score = round(accuracy_score(y_pred, y_test), 3)
# export decision tree ke dalam file dot    
export_graphviz(
    tree_model,
    out_file = "iris_tree.dot",
    feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica' ],
    rounded= True,
    filled =True)
print(f'Accuracy: {acc_score}')
# prediksi model dengan tree_model.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
print(tree_model.predict([[6.2, 3.4, 5.4, 2.3]])[0])

