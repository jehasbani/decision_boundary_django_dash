import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn import datasets


class Model:
    def __init__(self, name, **kwargs):
        classifiers = {
            "Nearest Neighbors": KNeighborsClassifier,
            'SVM': SVC,
            "Gaussian Process": GaussianProcessClassifier,
            "Decision Tree": DecisionTreeClassifier,
            "Random Forest": RandomForestClassifier,
            "Neural Net": MLPClassifier,
            "AdaBoost": AdaBoostClassifier,
            "Naive Bayes": GaussianNB,
            "LDA": LinearDiscriminantAnalysis,
            "QDA": QuadraticDiscriminantAnalysis}
        self.clf = classifiers[name](**kwargs)


class Dataset:
    def __init__(self, name, n_samples, noise):
        if name == "moons":
            self.X, self.y = datasets.make_moons(n_samples=n_samples,
                                                 noise=noise,
                                                 random_state=0)
        elif name == "circles":
            self.X, self.y = datasets.make_circles(n_samples=n_samples,
                                                   noise=noise,
                                                   factor=0.5,
                                                   random_state=1)
        elif name == "linear":
            X, y = datasets.make_classification(
                n_samples=n_samples,
                n_features=2,
                n_redundant=0,
                n_informative=2,
                random_state=2,
                n_clusters_per_class=1,
            )

            rng = np.random.RandomState(2)
            X += noise * rng.uniform(size=X.shape)

            self.X = X
            self.y = y
        else:
            raise ValueError(
                "Data type incorrectly specified. Please choose an existing dataset."
            )

