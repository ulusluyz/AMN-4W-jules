import torch
import torch.nn.functional as F
import random
import numpy as np

def run_decay_experiment(use_directional_update=False, use_orthogonal_keys=False, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    memory_size = 64
    S = torch.zeros(1, memory_size, memory_size)

    k_target = torch.randn(1, memory_size)
    k_target = F.normalize(k_target, p=2, dim=-1)
    v_target = torch.randn(1, memory_size)
    v_target = F.normalize(v_target, p=2, dim=-1)

    eta = torch.tensor([[1.0]])
    gamma = torch.tensor([[1.0]])

    if not use_directional_update:
        S_k = torch.bmm(S, k_target.unsqueeze(-1)).squeeze(-1)
        update_term = torch.bmm((v_target - S_k).unsqueeze(-1), k_target.unsqueeze(1))
        S = gamma.unsqueeze(-1) * S + eta.unsqueeze(-1) * update_term
    else:
        S_k = torch.bmm(S, k_target.unsqueeze(-1)).squeeze(-1)
        update_term = torch.bmm((eta * v_target - gamma * S_k).unsqueeze(-1), k_target.unsqueeze(1))
        S = S + update_term

    r_initial = torch.bmm(S, k_target.unsqueeze(-1)).squeeze(-1)

    for i in range(20):
        if use_orthogonal_keys:
            k_new = torch.randn(1, memory_size)
            k_new = k_new - torch.sum(k_new * k_target) * k_target
            k_new = F.normalize(k_new, p=2, dim=-1)
        else:
            k_new = torch.randn(1, memory_size)
            k_new = F.normalize(k_new, p=2, dim=-1)

        v_new = torch.randn(1, memory_size)
        v_new = F.normalize(v_new, p=2, dim=-1)

        eta_new = torch.tensor([[1.0]])
        gamma_new = torch.tensor([[0.95]])

        if not use_directional_update:
            S_k = torch.bmm(S, k_new.unsqueeze(-1)).squeeze(-1)
            update_term = torch.bmm((v_new - S_k).unsqueeze(-1), k_new.unsqueeze(1))
            S = gamma_new.unsqueeze(-1) * S + eta_new.unsqueeze(-1) * update_term
        else:
            S_k = torch.bmm(S, k_new.unsqueeze(-1)).squeeze(-1)
            update_term = torch.bmm((eta_new * v_new - gamma_new * S_k).unsqueeze(-1), k_new.unsqueeze(1))
            S = S + update_term

    r_final = torch.bmm(S, k_target.unsqueeze(-1)).squeeze(-1)

    sim = F.cosine_similarity(r_initial, r_final).item()
    mag_retention = (torch.norm(r_final) / torch.norm(r_initial)).item()
    return sim, mag_retention

print("--- Eski Formül (Global Gamma Decay) ---")
sim, mag = run_decay_experiment(False, False)
print(f"Rastgele Anahtarlar - CosSim: {sim:.3f}, Magnitude Retention: {mag:.3f}")
sim, mag = run_decay_experiment(False, True)
print(f"Ortogonal Anahtarlar - CosSim: {sim:.3f}, Magnitude Retention: {mag:.3f}")

print("\n--- Yeni Formül (Directional Forget Gate) ---")
sim, mag = run_decay_experiment(True, False)
print(f"Rastgele Anahtarlar - CosSim: {sim:.3f}, Magnitude Retention: {mag:.3f}")
sim, mag = run_decay_experiment(True, True)
print(f"Ortogonal Anahtarlar - CosSim: {sim:.3f}, Magnitude Retention: {mag:.3f}")
