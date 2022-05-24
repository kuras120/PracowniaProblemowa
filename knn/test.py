import pickle
import pandas as pd

from train import ddos_features


def ddos_test():
    model = pickle.load(open("./saved_model/ddos.sav", 'rb'))
    df1 = pd.read_csv('./knn_tuples_test.csv')
    X = df1.loc[:, ddos_features]
    it, acc, fp, fn = 0, 0, 0, 0
    for row in X.itertuples():
        list_row = list(row)
        list_row.pop(0)
        print(list_row)
        predicted = model.predict([list_row])
        actual = df1.iloc[[it]].label.values
        print('Predicted: ', predicted, 'Actual: ', actual)
        if predicted == actual:
            acc += 1
        elif predicted > actual:
            fp += 1
        else:
            fn += 1
        it += 1
    print('ACC: ', acc, ' FP: ', fp, ' FN: ', fn)
    f1_score = 2 * acc / (2 * acc + fp + fn)
    print(len(df1))
    acc = (acc / len(df1)) * 100
    print('Accuracy: ', acc)
    print('F1 score: ', f1_score)


if __name__ == "__main__":
    ddos_test()
