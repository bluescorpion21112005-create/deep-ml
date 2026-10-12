import torch

def qr_decomposition(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a = A.float()
    m, n = a.shape  # MANA BU YERDA TENGLIK BELGISI BO'LISHI KERAK

    q = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    r = torch.zeros((n, n), dtype=a.dtype, device=a.device)  # n, n bo'lishi kerak

    for j in range(n):
        v = a[:, j].clone()
        for i in range(j):
            r[i, j] = torch.dot(q[:, i], a[:, j])
            v = v - r[i, j] * q[:, i]

        r[j, j] = torch.norm(v)
        q[:, j] = v / r[j, j]

    return q, r