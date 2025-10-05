
import numpy as np
from src.core.j5_causal import _residualize, _ridge, j5_ahimsa_causal, j5_ahimsa_neuralscm

def test_causal_ridge_handles_confounder():
    rng = np.random.default_rng(0)
    N = 300
    C = rng.normal(size=(N,2))
    A = 0.9*C[:,0:1] + rng.normal(scale=0.2, size=(N,1))
    true_beta = np.array([0.5])
    noise = rng.normal(scale=0.2, size=(N,))
    harm = (A @ true_beta.reshape(-1,)).ravel() + 0.8*C[:,0] + noise

    corr = np.corrcoef(A.ravel(), harm.ravel())[0,1]
    assert corr > 0.6

    y_res = _residualize(harm, C)
    a_res = _residualize(A.ravel(), C)
    beta_hat = _ridge(a_res.reshape(-1,1), y_res, lam=1e-3)[0]
    assert abs(beta_hat - true_beta[0]) < 0.15

    J5 = j5_ahimsa_causal(
        H_i=np.random.randn(8,8), H_j=np.random.randn(8,8), decoder_j=None,
        A_hist=A, C_hist=C, harm_hist=harm, a_current=A.mean(axis=0), c_current=C.mean(axis=0),
        lam=1e-3, min_n=24
    )
    assert isinstance(J5, float)

def test_neural_scm_handles_nonlinearity():
    rng = np.random.default_rng(1)
    N = 600; dA, dC = 3, 4
    C = rng.normal(size=(N,dC))
    A = rng.normal(size=(N,dA)) + 0.5*np.sin(C[:,:1])
    true_fn = lambda a, c: 0.4*np.sin(a[:,0]) + 0.3*(a[:,1]*c[:,2]) - 0.2*(a[:,2]**2)
    harm = true_fn(A, C) + 0.5*(C[:,0]) + rng.normal(scale=0.1, size=(N,))

    idx = np.arange(N); rng.shuffle(idx)
    tr, te = idx[:500], idx[500:]
    A_tr, C_tr, y_tr = A[tr], C[tr], harm[tr]
    A_te, C_te, y_te = A[te], C[te], harm[te]

    j5 = j5_ahimsa_neuralscm(
        A_hist_cont=A_tr, C_hist=C_tr, harm_hist=y_tr,
        a_cur_cont=A_te.mean(axis=0), c_cur=C_te.mean(axis=0),
        min_n=64, epochs=60
    )
    assert j5 is not None and np.isfinite(j5)
