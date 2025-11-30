'''
Inputs (PER, Growth, Debt)
      │
MF Layer (27 fuzzy outputs)
      │
 ┌───────────────────────┬──────────────────────────┬─────────────────────────┐
 │ Product Rule (AND)    │ Additive Rule (SUM)      │ OR-like single rules    │
 └───────────────────────┴──────────────────────────┴─────────────────────────┘
      │
 Concatenate (Raw inputs도 포함)
      │
 Deep NN (ReLU, multi-layer)
      │
 Sugeno Output Layer (0~1 스케일)
'''

'''
그냥 딱 PER, 성장률, 부채비율 적당한 놈을 찾아내는 용도의 퍼지 신경망 모델.
근데 부채비율이 높을 수록 이상하게 너무 점수가 낮게 나옴.
50이상은 허용하지도 않음 ㅋㅋㅋ..
그래서 그냥 일단 DEBT 그냥 빼기..
x_mod[:,2] = 0.0 # DEBT 영향 제거
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianMF(nn.Module):
    def __init__(self, c, sigma):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float32))
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))

    def forward(self, x):
        return torch.exp(-0.5 * ((x - self.c) / (self.sigma + 1e-6)) ** 2)


class FuzzyLayer(nn.Module):
    def __init__(self, centers, sigmas):
        super().__init__()
        self.n_inputs = len(centers)
        self.n_mfs = len(centers[0])

        self.mfs = nn.ModuleList([
            nn.ModuleList([
                GaussianMF(centers[i][j], sigmas[i][j])
                for j in range(self.n_mfs)
            ])
            for i in range(self.n_inputs)
        ])

    def forward(self, x):
        outputs = []
        for i in range(self.n_inputs):
            inp = x[:, i].unsqueeze(1)
            mf_out = torch.cat([mf(inp) for mf in self.mfs[i]], dim=1)
            outputs.append(mf_out)
        return outputs


class ProductRuleLayer(nn.Module):
    def forward(self, fz):
        out = fz[0]
        for i in range(1, len(fz)):
            f = fz[i]
            B = out.size(0)
            if i == 2:
                f = torch.sqrt(f)
            out = out.unsqueeze(2) * f.unsqueeze(1)
            out = out.reshape(B, -1)
        return out / (out.sum(1, keepdim=True) + 1e-6)


class AdditiveRuleLayer(nn.Module):
    def forward(self, fz):
        return sum(fz)


class ORRuleLayer(nn.Module):
    def forward(self, fz):
        ors = [torch.max(f, dim=1, keepdim=True)[0] for f in fz]
        return torch.cat(ors, dim=1)


class HybridDeepNN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class FullFuzzyHybrid(nn.Module):
    def __init__(self):

        centers = [
            [0.15, 0.4, 0.75],
            [0.10, 0.35, 0.75],
            [0.20, 0.50, 0.85]
        ]

        sigmas = [
            [0.12, 0.15, 0.20],
            [0.05, 0.08, 0.12],
            [0.20, 0.25, 0.30]
        ]

        super().__init__()
        self.fuzzy = FuzzyLayer(centers, sigmas)

        self.prod_rule = ProductRuleLayer()
        self.add_rule = AdditiveRuleLayer()
        self.or_rule = ORRuleLayer()

        total_dim = (3**3) + 3 + 3 + 3
        self.nn = HybridDeepNN(total_dim)

    def forward(self, x):
        fz = self.fuzzy(x)

        prod_out = self.prod_rule(fz)
        add_out = self.add_rule(fz)
        or_out = self.or_rule(fz)

        x_mod = x.clone()
        x_mod[:,2] = 0.0  # DEBT 영향 제거

        full = torch.cat([x_mod, prod_out, add_out, or_out], dim=1)
        return self.nn(full)


if __name__ == "__main__":

    data = np.array([
        [5,20,30,0.85], [8,15,40,0.78], [10,10,50,0.65],
        [12,25,20,0.90], [15,30,10,0.95],
        [20,10,50,0.55], [25,15,60,0.50], [30,20,40,0.60],
        [35,25,30,0.70], [40,30,20,0.75],
        [50,20,60,0.40], [55,25,50,0.45], [60,30,40,0.50],
        [70,10,70,0.25], [80,15,65,0.28],
        [10,35,25,0.92], [15,40,20,0.95], [20,45,15,0.97],
        [25,50,10,0.99], [30,55,15,0.98],
        [12,5,80,0.30], [18,8,75,0.35], [22,12,70,0.40],
        [28,15,65,0.45], [35,18,60,0.50],
        [8,25,30,0.88], [10,28,25,0.91], [15,22,35,0.85],
        [18,26,40,0.78], [20,30,20,0.90]
    ], dtype=np.float32)

    X = torch.tensor(data[:, :3] / torch.tensor([80., 60., 100.]), dtype=torch.float32)
    Y = torch.tensor(data[:, 3:], dtype=torch.float32)

    model = FullFuzzyHybrid()
    optim = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.MSELoss()

    for epoch in range(3000):
        pred = model(X)
        loss = loss_fn(pred, Y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if epoch % 300 == 0:
            print(f"epoch {epoch} | loss={loss.item():.4f}")

    torch.save(model.state_dict(), "full_hybrid_model.pt")

    while True:
        print("\nEnter PER / Growth / Debt:")
        p = float(input("PER: "))
        g = float(input("Growth: "))
        # d = float(input("Debt: "))
        d = 0.0

        t = torch.tensor([[p/80, g/60, d/100]], dtype=torch.float32)
        out = model(t)
        print("Prediction:", out.item())
