import os
import numpy as np
import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier

ddos_features = ['src_ip_addr', 'src_port', 'dst_ip_addr', 'dst_port', 'protocol', 'f1', 'f2', 'f3', 'f4', 'f5',
                 'f6', 'f7', 'f8', 'f9', 'f10']


def train_ddos():
    if not os.path.exists('./knn_tuples_learn.csv') or not os.path.exists('./knn_tuples_test.csv'):
        df1 = pd.read_csv('./knn_tuples.csv')
        df1['src_ip_addr'] = df1['src_ip_addr'].map(lambda x: x.replace('.', '')).astype(int)
        df1['dst_ip_addr'] = df1['dst_ip_addr'].map(lambda x: x.replace('.', '')).astype(int)
        bad_samples = df1.iloc[:16395, :].sample(frac=1).reset_index(drop=True)
        good_samples = df1.iloc[16395:42720, :].sample(frac=1).reset_index(drop=True)
        print(len(bad_samples))
        print(len(good_samples))
        # 88 % of all rows
        learn = pd.concat([bad_samples.iloc[:14000, :], good_samples.iloc[:23000, :]]).sample(frac=1).reset_index(drop=True)
        learn.to_csv('./knn_tuples_learn.csv')
        # 12 % of all rows
        test = pd.concat([bad_samples.iloc[14000:16395, :], good_samples.iloc[23000:26325, :]]).sample(frac=1).reset_index(drop=True)
        test.to_csv('./knn_tuples_test.csv')
        # common rows check
        check = learn.merge(test, how='inner', indicator=False)
        print('Common rows between learn and test sets', check)
    else:
        learn = pd.read_csv('./knn_tuples_learn.csv')
        test = pd.read_csv('./knn_tuples_test.csv')
        check = learn.merge(test, how='inner', indicator=False)
        print('Common rows between learn and test sets', check)
    ddos_target = 'label'
    X = learn.loc[:, ddos_features]
    y = learn.loc[:, ddos_target]
    classes = np.unique(y)
    for i in range(len(classes)):
        if i == 2:
            learn = learn.replace(classes[i], 0)
        else:
            learn = learn.replace(classes[i], 1)
    print("Data preprocessing done.")
    model = KNeighborsClassifier(n_neighbors=3)
    # fitting our model
    model.fit(X.values, y)
    print("The model has been fit.")

    print("Save the fitted model?(y/n):")
    choice = input().lower()
    if choice == "y":
        pickle.dump(model, open("./saved_model/ddos.sav", 'wb'))


if __name__ == "__main__":
    train_ddos()
