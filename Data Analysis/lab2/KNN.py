from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Load datasets
breast_cancer = datasets.load_breast_cancer()
wine = datasets.load_wine()
digits = datasets.load_digits()

datasets = {
    "Breast Cancer Wisconsin": breast_cancer,
    "Wine": wine,
    "Optical recognition of handwritten digits": digits,
}

# k lang gieng
k_values = [1, 2, 3, 4, 5]

for name, dataset in datasets.items():
    X, y = dataset.data, dataset.target
    nFold = 5
    print(f"Dataset: {name}")
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, X, y, cv=nFold)  # kiem tra 5 fold
        print(f"k={k}, Accuracy: {scores.mean()*100}")
