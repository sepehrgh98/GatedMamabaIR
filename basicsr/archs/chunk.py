import torch
import torch.nn as nn
import torch.nn.functional as F


# Shapes used below: q, k [B,H,L,DK]; v [B,H,L,DV]; beta [B,H,L]; state S [B,H,DK,DV].
# log_alpha is [B,H,L] (one decay per head) or [B,H,L,DK] (one decay per key channel, a_t -> diag(a_t)).
# Recurrence (o_t = S_t^T q_t):
#   gated linear attention:  S_t = a_t S_{t-1} + b_t k_t v_t^T
#   gated delta rule:        S_t = (I - b_t k_t k_t^T) a_t S_{t-1} + b_t k_t v_t^T


def gated_delta_rule_recurrent(q, k, v, log_alpha, beta, use_delta=True, initial_state=None):
    """Token-by-token reference in fp32. Slow; only for testing chunk_gated_delta_rule."""
    q, k, v, beta = q.float(), k.float(), v.float(), beta.float()
    alpha = log_alpha.float().exp()
    if alpha.dim() == 3:
        alpha = alpha.unsqueeze(-1)  # per head -> broadcast over key channels
    B, H, L, DK = q.shape
    S = q.new_zeros(B, H, DK, v.shape[-1]) if initial_state is None else initial_state.float()
    outs = []
    for t in range(L):
        k_t, v_t, b_t = k[:, :, t], v[:, :, t], beta[:, :, t, None]
        S = alpha[:, :, t, :, None] * S
        if use_delta:
            v_t = v_t - torch.einsum('bhk,bhkv->bhv', k_t, S)  # erase what S already stores for k_t
        S = S + torch.einsum('bhk,bhv->bhkv', k_t, b_t * v_t)
        outs.append(torch.einsum('bhk,bhkv->bhv', q[:, :, t], S))
    return torch.stack(outs, dim=2), S


def chunk_gated_delta_rule(q, k, v, log_alpha, beta, chunk_size=64, use_delta=True, initial_state=None):
    """Chunked form of gated_delta_rule_recurrent, computed in fp32.

    Each chunk's output = read from the state of earlier chunks + causal attention inside the chunk.
    Decays are kept as log-cumsums and only exponentiated as differences (g_i - g_j <= 0), so they
    cannot underflow into 0/0. For the delta rule the within-chunk dependency is solved with the
    WY representation (a unit lower-triangular solve), as in Gated DeltaNet.

    Per-key-channel decay (log_alpha [B,H,L,DK]) cannot use a [T,T] decay matrix, so the pairwise
    decay is split around the chunk's middle token: (a_i e^{g_i - g_m}) . (b_j e^{g_m - g_j}).
    Each factor stays below e^{R/2}, where R is the chunk's total log-decay; callers must keep
    R <= ~160 (e.g. per-token log-decay >= -1 with chunk_size 128). Terms with j > i can overflow,
    but they are masked to 0 before use.

    Returns o [B,H,L,DV] (dtype of v) and the final state [B,H,DK,DV] (fp32).
    """
    out_dtype = v.dtype
    B, H, L, DK = q.shape
    DV = v.shape[-1]
    T = chunk_size
    per_channel = log_alpha.dim() == 4
    pad = (T - L % T) % T
    if pad:
        # zero keys/values with no decay leave the state untouched and come after every real token
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        log_alpha = F.pad(log_alpha, (0, 0, 0, pad) if per_channel else (0, pad))
        beta = F.pad(beta, (0, pad))
    N = (L + pad) // T

    q, k, v = (x.float().reshape(B, H, N, T, -1) for x in (q, k, v))
    beta = beta.float().reshape(B, H, N, T)
    # log of the decay since chunk start, as [B,H,N,T,1] (per head) or [B,H,N,T,DK] (per channel)
    g = log_alpha.float().reshape(B, H, N, T, -1).cumsum(-2)

    causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    if per_channel:
        g_mid = g[..., T // 2:T // 2 + 1, :]
        to_mid, from_mid = (g - g_mid).exp(), (g_mid - g).exp()

        def decayed_dot(a, b):  # [i, j] = sum_c a_ic b_jc prod_{j < s <= i} alpha_sc, for j <= i
            return ((a * to_mid) @ (b * from_mid).transpose(-1, -2)).masked_fill(~causal, 0)
    else:
        # decay[i, j] = prod_{j < s <= i} alpha_s for j <= i, else 0
        decay = (g - g.transpose(-1, -2)).masked_fill(~causal, float('-inf')).exp()

        def decayed_dot(a, b):
            return a @ b.transpose(-1, -2) * decay

    v_beta = v * beta.unsqueeze(-1)
    if use_delta:
        k_beta = k * beta.unsqueeze(-1)
        # (I + A) with A strictly lower: A[i, j] = b_i (k_i . k_j) decay[i, j]
        eye = torch.eye(T, device=q.device)
        system = eye + decayed_dot(k_beta, k).masked_fill(eye.bool(), 0)
        u = torch.linalg.solve_triangular(system, v_beta, upper=False, unitriangular=True)
        w = torch.linalg.solve_triangular(system, k_beta * g.exp(), upper=False, unitriangular=True)
    attn = decayed_dot(q, k)  # within-chunk scores, causal incl. the diagonal
    q_decayed = q * g.exp()
    k_to_end = k * (g[..., -1:, :] - g).exp()  # decay from each token to chunk end
    state_decay = g[..., -1:, :].exp().transpose(-1, -2)  # [B,H,N,DK or 1,1], scales rows of S

    S = q.new_zeros(B, H, DK, DV) if initial_state is None else initial_state.float()
    outs = []
    for n in range(N):
        v_new = u[:, :, n] - w[:, :, n] @ S if use_delta else v_beta[:, :, n]
        outs.append(q_decayed[:, :, n] @ S + attn[:, :, n] @ v_new)
        S = state_decay[:, :, n] * S + k_to_end[:, :, n].transpose(-1, -2) @ v_new
    o = torch.stack(outs, dim=2).reshape(B, H, N * T, DV)[:, :, :L]
    return o.to(out_dtype), S


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
    LEGACY, no longer used by the model: it has no read within the chunk and its decay can underflow.
    Kept so tests/test_gated_delta_scan.py can show the bug. Use chunk_gated_delta_rule instead.

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