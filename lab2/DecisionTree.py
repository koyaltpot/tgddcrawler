from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

datasets = {
    "Breast Cancer Wisconsin": datasets.load_breast_cancer(),
    "Wine": datasets.load_wine(),
    "Optical recognition of handwritten digits": datasets.load_digits(),
}

# Các tham số cho giải thuật Decision Tree
splitter = "best"
# splitter = 'best'
min_samples_split = 2

for name, dataset in datasets.items():
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = tree.DecisionTreeClassifier(
        splitter=splitter, min_samples_split=min_samples_split, criterion="gini"
    )
    scores = cross_val_score(clf, X, y, cv=10)

    print(f"Dataset: {name}")
    print(f"Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

    # Hiển thị cây
    # tree.plot_tree(clf.fit(X, y))
    # plt.show()
