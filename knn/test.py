import numpy as np
import sys
import pickle


def ddos_test(attributes):
    model = pickle.load(open("./saved_model/ddos.sav", 'rb'))
    result = model.predict([attributes])
    print(result)


if __name__ == "__main__":
    ddos_test(sys.argv[2:])
