import numpy as np
import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier


def train_ddos():
    df1 = pd.read_csv('./knn_tuples.csv')
    ddos_features = ['src_port', 'dst_port', 'protocol', 'f1', 'f2', 'f3', 'f4', 'f5',
                     'f6', 'f7', 'f8', 'f9', 'f10']
    ddos_target = 'label'
    X = df1.loc[:, ddos_features]
    y = df1.loc[:, ddos_target]
    classes = np.unique(y)
    for i in range(len(classes)):
        if i == 2:
            df1 = df1.replace(classes[i], 0)
        else:
            df1 = df1.replace(classes[i], 1)
    print("Data preprocessing done.")
    model = KNeighborsClassifier(n_neighbors=3)
    # fitting our model
    model.fit(X, y)
    print("The model has been fit.")

    print("Save the fitted model?(y/n):")
    choice = input().lower()
    if choice == "y":
        pickle.dump(model, open("./saved_model/ddos.sav", 'wb'))


if __name__ == "__main__":
    train_ddos()
