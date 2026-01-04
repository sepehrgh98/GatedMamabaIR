import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_gamma(alpha, stable_gamma=True):
    """
    alpha: [B,H,T] (any dtype)
    returns gamma: [B,H,T] fp32
    """
    a32 = alpha.float().clamp_min(1e-6)
    if stable_gamma:
        return torch.exp(torch.cumsum(torch.log(a32), dim=2))  # fp32
    else:
        return torch.cumprod(a32, dim=2)


def chunk_gated_delta_fullrank(Q, K, V, alpha, beta, S, stable_gamma=True, eps=1e-12):
    """
    Full-rank state version.
    Q,K,V : bf16
    alpha,beta : [B,H,T]
    S : [B,H,DV,DK] bf16
    """
    gamma = _compute_gamma(alpha, stable_gamma=stable_gamma)   # fp32

    # ---- READ ----
    gamma_q = gamma.to(Q.dtype)
    O = torch.matmul(Q * gamma_q.unsqueeze(-1), S.transpose(-1, -2))  # [B,H,T,DV]

    # ---- WRITE ----
    gamma_T = gamma[:, :, -1]  # fp32
    scale = gamma_T.unsqueeze(-1) / gamma.clamp_min(eps)  # fp32

    Vhat = V * beta.to(V.dtype).unsqueeze(-1)
    V_scaled = Vhat * scale.to(V.dtype).unsqueeze(-1)

    dS = torch.matmul(V_scaled.transpose(-1, -2), K)  # [B,H,DV,DK]
    S_next = gamma_T.to(S.dtype).unsqueeze(-1).unsqueeze(-1) * S + dS

    return O, S_next


def chunk_gated_delta_lowrank(Q, K, V, alpha, beta, U, W, g, stable_gamma=True, eps=1e-12):
    """
    Low-rank state: S ≈ U W^T

    Q,K,V : bf16
    U : [B,H,DV,r] bf16
    W : [B,H,DK,r] bf16
    g : [B,H,T,r]  bf16  (token gate for rank space, computed from K)
    """
    gamma = _compute_gamma(alpha, stable_gamma=stable_gamma)   # fp32
    gamma_q = gamma.to(Q.dtype)

    # ---- READ ----
    # z = (Q * gamma) @ W   -> [B,H,T,r]
    z = torch.matmul(Q * gamma_q.unsqueeze(-1), W)  # W is [B,H,DK,r]
    # O = z @ U^T -> [B,H,T,DV]
    O = torch.matmul(z, U.transpose(-1, -2))

    # ---- WRITE ----
    gamma_T = gamma[:, :, -1]  # fp32
    scale = gamma_T.unsqueeze(-1) / gamma.clamp_min(eps)  # fp32

    Vhat = V * beta.to(V.dtype).unsqueeze(-1)
    V_scaled = Vhat * scale.to(V.dtype).unsqueeze(-1)  # [B,H,T,DV]

    # Update U using rank gates g:
    # dU = V_scaled^T @ g   : (DV,T)@(T,r) -> (DV,r)
    dU = torch.matmul(V_scaled.transpose(-1, -2), g)  # [B,H,DV,r]

    # Update W using keys and the same gates:
    # dW = K^T @ g : (DK,T)@(T,r) -> (DK,r)
    dW = torch.matmul(K.transpose(-1, -2), g)  # [B,H,DK,r]

    decay = gamma_T.to(U.dtype).unsqueeze(-1).unsqueeze(-1)
    U_next = decay * U + dU
    W_next = decay * W + dW

    return O, U_next, W_next