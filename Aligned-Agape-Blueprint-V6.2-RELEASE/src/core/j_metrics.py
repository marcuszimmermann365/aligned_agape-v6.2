"""J-metrics module: J1–J4 and legacy Proxy-J5 (baseline).
Causal J5 implementations live in src/core/j5_causal.py and are used by default via CLI --j5.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def j1_integration(H, eps=1e-6):
    H = np.asarray(H)
    Hc = H - H.mean(axis=0, keepdims=True)
    cov = (Hc.T @ Hc) / max(1, (H.shape[0]-1))
    tr = np.trace(cov)
    if tr <= eps: tr = eps
    cov_norm = cov / tr
    sign, logdet = np.linalg.slogdet(cov_norm + eps * np.eye(cov.shape[0]))
    return float(logdet)

def j2_diversity(H):
    H = np.asarray(H)
    Hc = H - H.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(Hc, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0.0
    return float(-np.var(s))

def finite_diff_grad_norm(f, z, eps=1e-3):
    z = np.asarray(z, dtype=float)
    gradsq = 0.0
    base = f(z)
    for i in range(z.shape[0]):
        dz = np.zeros_like(z); dz[i] = eps
        plus = f(z + dz); minus = f(z - dz)
        d = (plus - minus) / (2*eps)
        gradsq += d**2
    return -float(gradsq)

def j3_robustness(f_recon_scalar, z):
    return finite_diff_grad_norm(f_recon_scalar, z)

def j4_empowerment_infonce(actions, s_next, s_cond, hidden=64, steps=80):
    A = torch.as_tensor(actions, dtype=torch.float32)
    S_next = torch.as_tensor(s_next, dtype=torch.float32)
    S_cond = torch.as_tensor(s_cond, dtype=torch.float32)
    B = A.size(0)
    enc_ctx = nn.Sequential(nn.Linear(A.size(1)+S_cond.size(1), hidden), nn.ReLU(),
                            nn.Linear(hidden, hidden), nn.ReLU())
    enc_next = nn.Sequential(nn.Linear(S_next.size(1), hidden), nn.ReLU(),
                             nn.Linear(hidden, hidden), nn.ReLU())
    score = nn.Linear(hidden, 1, bias=False)
    params = list(enc_ctx.parameters()) + list(enc_next.parameters()) + list(score.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    for _ in range(steps):
        ctx = torch.cat([A, S_cond], dim=-1)
        h_ctx = enc_ctx(ctx)
        h_next = enc_next(S_next)
        proj = F.linear(h_next, score.weight)
        logits = h_ctx @ proj.t()
        labels = torch.arange(B)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ctx = torch.cat([A, S_cond], dim=-1)
        h_ctx = enc_ctx(ctx); h_next = enc_next(S_next)
        proj = F.linear(h_next, score.weight)
        logits = h_ctx @ proj.t()
        mi = torch.log(torch.tensor(B, dtype=torch.float32)) - F.cross_entropy(logits, torch.arange(B))
        return float(mi.item())

def j4_empowerment_proxy(outputs):
    outputs = np.asarray(outputs)
    var = outputs.var(axis=0).sum() if outputs.ndim > 1 else outputs.var()
    return float(np.log(var + 1e-8) if var > 0 else -50.0)

def j5_ahimsa_proxy(H_i, H_j, decoder_j):
    xj = decoder_j(H_j)
    xj_pert = decoder_j(H_j + H_i)
    harm = np.mean((xj_pert - xj)**2)
    return float(-harm)

def procrustes_distance(H1, H2, reg=1e-4):
    H1c = H1 - H1.mean(axis=0, keepdims=True)
    H2c = H2 - H2.mean(axis=0, keepdims=True)
    M = H2c.T @ H1c
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    H1a = H1c @ R.T
    dist = np.mean((H1a - H2c)**2) + reg * np.sum((R - np.eye(R.shape[0]))**2)
    return float(dist)

def social_coherence(H_list):
    if len(H_list) < 2: return 1.0
    dists = []
    for i in range(len(H_list)):
        for j in range(i+1, len(H_list)):
            dists.append(procrustes_distance(H_list[i], H_list[j]))
    d = np.mean(dists)
    return float(1.0 / (1.0 + d))
