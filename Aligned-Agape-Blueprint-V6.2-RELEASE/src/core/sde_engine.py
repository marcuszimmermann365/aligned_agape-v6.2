
import torch, torch.nn as nn

class NeuralPotential(nn.Module):
    def __init__(self, q_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(q_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, q):
        return self.net(q).squeeze(-1)

class HamiltonianSDE(nn.Module):
    def __init__(self, q_dim, mass=1.0, sigma=0.0, potential=None):
        super().__init__()
        self.q_dim = q_dim
        self.mass = mass
        self.inv_mass = 1.0/mass
        self.sigma = sigma
        self.V = potential if potential is not None else NeuralPotential(q_dim)
    def H(self, q, p):
        return 0.5 * (p**2).sum(dim=-1) * self.inv_mass + self.V(q)

class SymplecticIntegrator:
    def __init__(self, sde: HamiltonianSDE):
        self.sde = sde
    @torch.no_grad()
    def step(self, q, p, dt):
        q.requires_grad_(True)
        V = self.sde.V(q).sum()
        gradV = torch.autograd.grad(V, q)[0]
        p = p - 0.5*dt*gradV
        q = q + dt * p * self.sde.inv_mass
        q.requires_grad_(True)
        V2 = self.sde.V(q).sum()
        gradV2 = torch.autograd.grad(V2, q)[0]
        p = p - 0.5*dt*gradV2
        if self.sde.sigma>0:
            p = p + self.sde.sigma * torch.sqrt(torch.tensor(dt)) * torch.randn_like(p)
        return q, p
