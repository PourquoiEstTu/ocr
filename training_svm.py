import os
import numpy as np
import preprocessing as prep
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DIR = "/windows/Users/thats/Documents/ocr-repo-files"
DATA = "dataset2/Img"
FEATURE_DIR = f"{DIR}/features"

def get_X_y_data(num_of_files: int, total_files:int, input_dir: str, path_to_label: str) -> tuple[np.ndarray, np.ndarray] :
    if num_of_files <= 0 :
        raise ValueError("Please enter a positive number as the first argument")
    rng = np.random.default_rng()
    X = []
    y = []
    for _ in range(num_of_files) :
        count = 0
        idx = rng.integers(0, total_files)
        for file in os.scandir(input_dir) :
            if count == idx :
                # print(count)
                # print(idx)
                # no bounds checking on this function, so could error, try catch
                #   block is the workaround for now
                try :
                    feature = np.load(f"{input_dir}/{file.name}")
                    label = np.load(f"{path_to_label}")[idx]
                except IndexError :
                    continue
                # print(feature)
                # print(label)
                X.append(feature)
                y.append(label)
                break
            else :
                count += 1
    X = np.array(X)
    y = np.array(y)
    return X, y

    # print(X.shape)

class MulticlassSvm :
    def __init__(self, classifier: str) :
        if classifier == "SVC" :
            self.clf = svm.SVC(decision_function_shape='ovo')
        elif classifier == "NuSVC" :
            self.clf = svm.NuSVC(decision_function_shape='ovo')
        else :
            raise Exception("Please give 'SVC' or 'NuSVC' as input to class")

    def fit(self, X: np.ndarray, y:np.ndarray) :
        self.clf.fit(X, y)

    def predict(self, x: np.ndarray) :
        return self.clf.predict(x)

# def fit_SVC(X: np.ndarray, y:np.ndarray) :
#     clf.fit

def main() :
    # init data
    # i can't use all 3411 files b/c I don't have enough ram
    num_of_files = 100
    X,y = get_X_y_data(num_of_files, 3410, FEATURE_DIR, f"{FEATURE_DIR}/ordered_labels.npy")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                                random_state=42)
    print("here1")
    # convert y's letters to numerals
    # le_train = LabelEncoder()
    # le_test = LabelEncoder()
    # y_train_numeric = le_train.fit_transform(y_train)
    # y_test_numeric = le_test.fit_transform(y_test)
    print("here2")
    # print(le_train.classes_)
    # print(y_train_numeric)
    # print(le_train.inverse_transform([1]))

    svc = MulticlassSvm("SVC")
    print(y_train)
    print(y_test)
    svc.fit(X_train, y_train)
    print("here3")
    prediction = svc.predict([X_test[0]])#.astype(np.int64)
    print(f"X_test[0] prediction = {prediction}")
    print(f"y_test[0] true label = {y_test[0]}")
    # print(f"label given by svm: {le_train.inverse_transform(prediction)}")
    # print(le_train.classes_)
main()
