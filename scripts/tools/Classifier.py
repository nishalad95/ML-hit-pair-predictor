import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity

class KDEClassifier(BaseEstimator, ClassifierMixin):
    """Bayesian generative classification based on KDE
    
    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    """
    
    
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
    
    
    def fit(self, X, y):
        # needed for probability calibration
        #if str(type(X)) != "<class 'pandas.core.frame.DataFrame'>":
        #    X = np.reshape(X, (len(X), 1)).T
        #    y = np.reshape(y, (len(y), 1)).T
        #    X = pd.DataFrame({'|cot(t)|': X[0]})
        #    y = pd.DataFrame({'class': y[0]})
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X.iloc[list(np.where(y == yi)[0])] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth, 
                                      kernel=self.kernel).fit(Xi) for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0]) for Xi in training_sets]
        return self
    
    
    def predict_proba(self, X):
        logprobs = np.array([model.score_samples(X) for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        self.predicted_proba_ = result / result.sum(1, keepdims=True)
        return self.predicted_proba_
    
    
    def predict(self, X, thres):
        #return self.classes_[np.argmax(self.predict_proba(X), 1)]
        class_probabilities = self.predict_proba(X)[:, 1]
        classes = (class_probabilities > thres).astype(int)
        return classes

