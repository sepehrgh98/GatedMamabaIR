

def l2norm(x, dim=-1, eps=1e-6):
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)
