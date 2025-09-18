
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, Sequence
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from .utils import l2_normalize

MethodName = Literal["dom","logistic","hinge"]
method_names: Tuple[str,...] = ("dom","logistic","hinge")

@dataclass
class Concept:
    name: str
    tensor: "torch.Tensor" = None

def _train_logistic(X: np.ndarray, y: np.ndarray, C: float=1.0, max_iter: int=2000, random_state: int=0):
    lr = LogisticRegression(penalty="l2", C=C, solver="liblinear", max_iter=max_iter, random_state=random_state)
    lr.fit(X,y)
    w = lr.coef_.ravel()
    b = float(lr.intercept_.ravel()[0])
    acc = float((lr.predict(X)==y).mean())
    return w,b,acc

def _train_hinge(X: np.ndarray, y: np.ndarray, alpha: float=1e-4, max_iter: int=3000, random_state: int=0):
    svm = SGDClassifier(loss="hinge", alpha=alpha, max_iter=max_iter, tol=1e-3, random_state=random_state)
    svm.fit(X,y)
    w = svm.coef_.ravel()
    b = float(svm.intercept_.ravel()[0]) if svm.fit_intercept else 0.0
    acc = float((svm.predict(X)==y).mean())
    return w,b,acc

def _train_dom(Xp: np.ndarray, Xn: np.ndarray):
    mu_p = Xp.mean(axis=0); mu_n = Xn.mean(axis=0)
    w = mu_p - mu_n
    b = -0.5*float((mu_p+mu_n)@w)
    X = np.vstack([Xp,Xn]); y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))])
    acc = float((((X@w)+b>=0).astype(int)==y).mean())
    return w,b,acc

def train_cav(X_pos: np.ndarray, X_neg: np.ndarray, method: MethodName="dom", random_state: int=0, **kwargs):
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    if method=="dom":
        w,b,acc = _train_dom(X_pos, X_neg)
    elif method=="logistic":
        w,b,acc = _train_logistic(X,y, **kwargs, random_state=random_state)
    elif method=="hinge":
        w,b,acc = _train_hinge(X,y, **kwargs, random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")
    w = l2_normalize(w)
    return {"vector": w, "bias": float(b), "acc": float(acc), "method": method, "meta": {"random_state": random_state}}

def sample_train_cav(X_pos_all: np.ndarray, X_neg_all: np.ndarray, n_examples: int, method: MethodName, random_state: int=0, with_replacement: bool=False, **kwargs):
    rng = np.random.default_rng(random_state)
    def _sample(X):
        idx = rng.choice(len(X), size=min(n_examples, len(X)) if not with_replacement else n_examples, replace=with_replacement)
        return X[idx]
    return train_cav(_sample(X_pos_all), _sample(X_neg_all), method=method, random_state=random_state, **kwargs)
