
import numpy as np
from typing import List, Tuple, Optional
from .j_metrics import j5_ahimsa_proxy

# ---- Helpers: actions & context derived minimally from actors/state ----
def action_from_actor(actor):
    # Use latent mean as action proxy (continuous)
    return np.asarray(actor.hidden.mean(axis=0), dtype=float)

def build_context_features(actors, state):
    feats = []
    for ag in actors:
        h = np.asarray(ag.hidden)
        feats.extend([float(np.mean(h)), float(np.std(h))])
    feats.append(1.0 if state.get("esc","Tension")=="Escalation" else 0.0)
    return np.asarray(feats, dtype=float)

# ---- Ridge-orthogonalized causal J5 ----
def _ridge(X, y, lam=1e-3):
    XT = X.T
    A = XT @ X + lam * np.eye(X.shape[1])
    b = XT @ y
    return np.linalg.solve(A, b)

def _residualize(Y, C, lam=1e-3):
    Y = np.asarray(Y).reshape(-1)
    C = np.asarray(C)
    if C.ndim == 1: C = C[:, None]
    C1 = np.hstack([C, np.ones((C.shape[0], 1))])
    w = _ridge(C1, Y, lam)
    Y_hat = C1 @ w
    return Y - Y_hat

def j5_ahimsa_causal(
    H_i, H_j, decoder_j,
    A_hist, C_hist, harm_hist,
    a_current, c_current,
    lam=1e-3, min_n=24
):
    if A_hist is None or C_hist is None or harm_hist is None: return None
    A = np.asarray(A_hist); C = np.asarray(C_hist); y = np.asarray(harm_hist).reshape(-1)
    if len(A) < min_n or len(C)!=len(A) or len(y)!=len(A): return None
    y_res = _residualize(y, C, lam=lam)
    A_res = np.zeros_like(A, dtype=float)
    for k in range(A.shape[1] if A.ndim>1 else 1):
        col = A[:, k] if A.ndim>1 else A.reshape(-1)
        A_res[:, k if A.ndim>1 else 0] = _residualize(col, C, lam=lam)
    w = _ridge(A_res, y_res, lam=lam)
    a_cur = np.asarray(a_current).reshape(1, -1)
    c_cur = np.asarray(c_current).reshape(1, -1)
    # residualize current action against context
    A_ext = np.vstack([A, a_cur])
    C_ext = np.vstack([C, c_cur])
    a_cur_res = np.zeros_like(a_cur)
    for k in range(a_cur.shape[1]):
        col_ext = np.concatenate([A[:,k], a_cur[:,k]]) if A.ndim>1 else np.concatenate([A.reshape(-1), a_cur.reshape(-1)])
        a_cur_res[:, k] = _residualize(col_ext, C_ext, lam=lam)[-1]
    harm_effect = float(a_cur_res @ w.reshape(-1,1))
    return float(-max(0.0, harm_effect))

# ---- AIPW (Doubly-Robust) for discrete actions (pure-numpy logistic) ----
def _softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def _logistic_fit_multiclass(C, A_disc, max_iter=300, lr=0.1, reg=1e-3, K=None):
    # simple softmax regression (one-vs-rest style)
    N, d = C.shape
    if K is None: K = int(np.max(A_disc))+1
    W = np.zeros((d, K))
    b = np.zeros((K,))
    Y = np.eye(K)[A_disc]
    for _ in range(max_iter):
        scores = C @ W + b
        P = _softmax(scores)
        grad_W = C.T @ (P - Y) / N + reg * W
        grad_b = np.mean(P - Y, axis=0)
        W -= lr * grad_W
        b -= lr * grad_b
    return W, b

def _ridge_fit(X, y, lam=1e-3):
    XT = X.T
    A = XT @ X + lam * np.eye(X.shape[1])
    b = XT @ y
    return np.linalg.solve(A, b)

def j5_ahimsa_aipw(A_hist_disc, C_hist, harm_hist, a_cur_disc, c_cur, min_n=64):
    if A_hist_disc is None or C_hist is None or harm_hist is None: return None
    A = np.asarray(A_hist_disc).astype(int).reshape(-1)
    C = np.asarray(C_hist); y = np.asarray(harm_hist).reshape(-1)
    if len(A) < min_n or len(C)!=len(A) or len(y)!=len(A): return None
    # policy model π(a|C)
    W, b = _logistic_fit_multiclass(C, A, max_iter=300, lr=0.2, reg=1e-4)
    scores = C @ W + b
    P = _softmax(scores)
    # outcome model μ(a,C) via one-hot
    K = int(np.max(A))+1
    A1h = np.eye(K)[A]
    Xo = np.hstack([C, A1h])
    w = _ridge_fit(Xo, y, lam=1e-3)
    # score at current
    a = int(a_cur_disc)
    c = np.asarray(c_cur).reshape(1,-1)
    a1h = np.eye(K)[[a]]
    mu = float(np.hstack([c, a1h]) @ w)
    pi = float(_softmax(c @ W + b)[0,a])
    pi = max(pi, 1e-6)
    psi = mu  # no observed y at inference time
    return float(-psi)

# ---- Neural-SCM (NumPy MLP) ----
class _MLP:
    def __init__(self, in_dim, hidden=64, out_dim=1, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(scale=0.1, size=(in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(scale=0.1, size=(hidden, out_dim))
        self.b2 = np.zeros(out_dim)

    def forward(self, X):
        H = np.tanh(X @ self.W1 + self.b1)
        Y = H @ self.W2 + self.b2
        return Y, H

    def step(self, X, y, lr=1e-3, l2=1e-4, sample_weight=None):
        y = y.reshape(-1,1)
        Y, H = self.forward(X)
        if sample_weight is None:
            grad = (Y - y)
        else:
            sw = sample_weight.reshape(-1,1)
            grad = sw * (Y - y)
        dW2 = H.T @ grad + l2*self.W2
        db2 = grad.sum(axis=0)
        dH  = grad @ self.W2.T * (1 - H**2)
        dW1 = X.T @ dH + l2*self.W1
        db1 = dH.sum(axis=0)
        self.W2 -= lr * dW2; self.b2 -= lr * db2
        self.W1 -= lr * dW1; self.b1 -= lr * db1
        return float(np.mean((Y - y)**2))

class _NeuralSCM:
    def __init__(self, in_dim_c, in_dim_a, hidden=64, seed=0):
        self.mu = _MLP(in_dim=in_dim_c+in_dim_a, hidden=hidden, out_dim=1, seed=seed)

    def fit(self, A, C, y, epochs=200, lr=1e-3, l2=1e-4, iw=None):
        A = np.asarray(A); C = np.asarray(C); y = np.asarray(y).reshape(-1)
        X = np.hstack([C, A])
        for _ in range(epochs):
            self.mu.step(X, y, lr=lr, l2=l2, sample_weight=iw)

    def predict(self, a, c):
        Xc = np.hstack([np.asarray(c).reshape(1,-1), np.asarray(a).reshape(1,-1)])
        pred, _ = self.mu.forward(Xc)
        return float(pred.ravel()[0])

def j5_ahimsa_neuralscm(A_hist_cont, C_hist, harm_hist, a_cur_cont, c_cur, min_n=64, epochs=200, iw_clip=10.0):
    if A_hist_cont is None or C_hist is None or harm_hist is None: return None
    A = np.asarray(A_hist_cont); C = np.asarray(C_hist); y = np.asarray(harm_hist).reshape(-1)
    if len(A) < min_n or len(C)!=len(A) or len(y)!=len(A): return None
    # Simple Gaussian IW (stabilized)
    try:
        mu = A.mean(axis=0, keepdims=True)
        cov = np.cov(A.T) + 1e-6*np.eye(A.shape[1])
        invcov = np.linalg.inv(cov)
        def q(a):
            d = (a-mu) @ invcov @ (a-mu).T
            return float(np.exp(-0.5*d.squeeze()))
        iw = np.array([1.0/max(q(A[i:i+1,:]), 1e-6) for i in range(len(A))])
        iw = np.clip(iw, 0.0, iw_clip)
    except Exception:
        iw = None
    model = _NeuralSCM(in_dim_c=C.shape[1], in_dim_a=A.shape[1], hidden=64, seed=0)
    model.fit(A, C, y, epochs=epochs, lr=1e-3, l2=1e-4, iw=iw)
    pred = model.predict(a_cur_cont, c_cur)
    return float(-pred)

# ---- Unified compute with fallbacks ----
def compute_j5_with_fallbacks(mode:str, state, actors, a_cont, a_disc, C_feat):
    A_cont = np.asarray(state["replay_A_cont"]) if len(state["replay_A_cont"]) else None
    A_disc = np.asarray(state["replay_A_disc"]) if len(state["replay_A_disc"]) else None
    C_hist = np.asarray(state["replay_C"]) if len(state["replay_C"]) else None
    y_hist = np.asarray(state["replay_harm"]) if len(state["replay_harm"]) else None

    def ridge():
        return j5_ahimsa_causal(
            H_i=actors[0].hidden, H_j=actors[1].hidden, decoder_j=actors[1].decoder,
            A_hist=A_cont, C_hist=C_hist, harm_hist=y_hist,
            a_current=a_cont, c_current=C_feat, min_n=24
        )
    def aipw():
        return j5_ahimsa_aipw(A_hist_disc=A_disc, C_hist=C_hist, harm_hist=y_hist,
                              a_cur_disc=a_disc, c_cur=C_feat, min_n=64)
    def neural():
        return j5_ahimsa_neuralscm(A_hist_cont=A_cont, C_hist=C_hist, harm_hist=y_hist,
                                   a_cur_cont=a_cont, c_cur=C_feat, min_n=64)

    tried = []
    if mode == "neural":
        v = neural(); 
        if v is not None: return v, None
        tried.append("neural")
        v = aipw(); 
        if v is not None: return v, "aipw"
        tried.append("aipw")
        v = ridge(); 
        if v is not None: return v, "ridge"
        tried.append("ridge")
        return j5_ahimsa_proxy(actors[0].hidden, actors[1].hidden, actors[1].decoder), "proxy"

    if mode == "aipw":
        v = aipw(); 
        if v is not None: return v, None
        tried.append("aipw")
        v = neural(); 
        if v is not None: return v, "neural"
        tried.append("neural")
        v = ridge(); 
        if v is not None: return v, "ridge"
        tried.append("ridge")
        return j5_ahimsa_proxy(actors[0].hidden, actors[1].hidden, actors[1].decoder), "proxy"

    if mode == "ridge":
        v = ridge(); 
        if v is not None: return v, None
        tried.append("ridge")
        v = neural(); 
        if v is not None: return v, "neural"
        tried.append("neural")
        v = aipw(); 
        if v is not None: return v, "aipw"
        tried.append("aipw")
        return j5_ahimsa_proxy(actors[0].hidden, actors[1].hidden, actors[1].decoder), "proxy"

    # proxy
    return j5_ahimsa_proxy(actors[0].hidden, actors[1].hidden, actors[1].decoder), None
