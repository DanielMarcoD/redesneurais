"""Insper 2026.2: dois VAEs MNIST, seis figuras e resultados reais em JSON.

Executar: python docs/exercises/vae/code/vae.py
O download fica em data/MNIST (ignorado pelo Git); nenhum checkpoint é salvo.
"""

import json
import os
from pathlib import Path
import platform
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import scipy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parents[1]
SEED, EPOCHS, BATCH_SIZE, LEARNING_RATE = 42, 20, 128, 1e-3


class VAE(nn.Module):
    """Mesma arquitetura em ambos os exercícios; apenas d muda."""

    def __init__(self, d=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU()
        )
        self.mu = nn.Linear(256, d)
        self.logvar = nn.Linear(256, d)
        self.decoder = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x.flatten(1))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        # Ruído independente dos parâmetros; mu e std continuam diferenciáveis.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


def loss_terms(reconstruction, x, mu, logvar):
    """BCE + KL analítica, somadas por imagem e médias no batch; beta = 1."""
    x = x.flatten(1)
    assert reconstruction.shape == x.shape and x.shape[1] == 784
    assert mu.shape == logvar.shape and mu.shape[0] == x.shape[0]
    bce = F.binary_cross_entropy(reconstruction, x, reduction="sum") / len(x)
    kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum() / len(x)
    return bce + kl, bce, kl


def validate_implementation():
    """Conferências independentes, sem consumir a seed dos experimentos."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        x = torch.rand(3, 1, 28, 28)
        prediction = torch.full((3, 784), 0.25, dtype=torch.float64,
                                requires_grad=True)
        mu = torch.zeros(3, 2, dtype=torch.float64, requires_grad=True)
        logvar = torch.zeros_like(mu, requires_grad=True)
        total, bce, kl = loss_terms(prediction, x.double(), mu, logvar)
        # BCE calculada diretamente pelos logaritmos, não pelo helper PyTorch.
        expected = -(x.double().flatten(1) * prediction.log()
                     + (1 - x.double().flatten(1)) * torch.log1p(-prediction))
        torch.testing.assert_close(bce, expected.sum(1).mean())
        assert kl.item() == 0
        shifted = loss_terms(prediction, x.double(), mu + 1, logvar)[2]
        torch.testing.assert_close(shifted, torch.tensor(1., dtype=torch.float64))
        variance_kl = loss_terms(prediction, x.double(), mu,
                                logvar + np.log(4))[2]
        torch.testing.assert_close(variance_kl, torch.tensor(
            4 - 1 - np.log(4), dtype=torch.float64))
        # Repetir imagens não pode mudar a média por imagem, inclusive a KL.
        repeated = loss_terms(prediction.repeat(2, 1), x.double().repeat(2, 1, 1, 1),
                              (mu + 1).repeat(2, 1), (logvar + np.log(4)).repeat(2, 1))
        original = loss_terms(prediction, x.double(), mu + 1, logvar + np.log(4))
        for a, b in zip(repeated, original):
            torch.testing.assert_close(a, b)
        assert torch.autograd.gradcheck(
            lambda p, m, v: loss_terms(p, x.double(), m, v)[0],
            (prediction, mu, logvar), fast_mode=True
        )
        for d in (16, 2):
            model = VAE(d)
            reconstruction, m, v = model(x)
            assert reconstruction.shape == (3, 784) and m.shape == v.shape == (3, d)
            assert torch.isfinite(reconstruction).all()
            assert ((reconstruction >= 0) & (reconstruction <= 1)).all()
            # Verificar a fórmula e as derivadas com o mesmo epsilon.
            m = m.detach().requires_grad_()
            v = v.detach().requires_grad_()
            torch.manual_seed(123)
            epsilon = torch.randn_like(m)
            torch.manual_seed(123)
            z = model.reparameterize(m, v)
            torch.testing.assert_close(z, m + (0.5 * v).exp() * epsilon)
            gm, gv = torch.autograd.grad(z.sum(), (m, v))
            torch.testing.assert_close(gm, torch.ones_like(m))
            torch.testing.assert_close(gv, 0.5 * (0.5 * v).exp() * epsilon)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            before = model.encoder[0].weight.detach().clone()
            reconstruction, m, v = model(x)
            loss_terms(reconstruction, x, m, v)[0].backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.parameters())
            optimizer.step()
            assert not torch.equal(before, model.encoder[0].weight)
    return {"BCE_logarithms": True, "KL_zero_shift_variance": True,
            "batch_reduction": True, "loss_gradcheck": True,
            "shapes_range": True, "reparameterization_gradients": True,
            "finite_gradients_parameter_update": True}


def run_epoch(model, loader, device, optimizer=None):
    """Validação/teste não alteram pesos; os rótulos são ignorados."""
    training = optimizer is not None
    model.train(training)
    before = [p.detach().clone() for p in model.parameters()] if not training else None
    totals = np.zeros(3)
    count = 0
    with torch.set_grad_enabled(training):
        for images, _ in loader:
            x = images.to(device)
            reconstruction, mu, logvar = model(x)
            assert reconstruction.requires_grad == training
            terms = loss_terms(reconstruction, x, mu, logvar)
            assert torch.isfinite(torch.stack(terms)).all(), "Loss não finita"
            if training:
                optimizer.zero_grad()
                terms[0].backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters())
                optimizer.step()
            totals += np.array([v.item() for v in terms]) * len(x)
            count += len(x)
    if not training:
        assert all(torch.equal(a, b) for a, b in zip(before, model.parameters())), \
            "Pesos alterados na avaliação"
    return dict(zip(("total", "reconstruction", "kl"), (totals / count).tolist()))


def train_model(d, train, val, device):
    # Reset antes de CADA modelo; mesma ordem de batches nas duas dimensões.
    torch.manual_seed(SEED)
    model = VAE(d).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val, batch_size=BATCH_SIZE,
                            generator=torch.Generator().manual_seed(SEED))
    history = []
    for epoch in range(1, EPOCHS + 1):
        training = run_epoch(model, train_loader, device, optimizer)
        validation = run_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train": training, "validation": validation})
        print(f"d={d:2} época={epoch:02} treino={training['total']:.3f} "
              f"val={validation['total']:.3f} BCE={validation['reconstruction']:.3f} "
              f"KL={validation['kl']:.3f}", file=sys.stderr, flush=True)
    assert history[-1]["validation"]["total"] < history[0]["validation"]["total"]
    return model, {"parameters": sum(p.numel() for p in model.parameters()),
                   "history": history}


def save_figure(fig, number):
    fig.savefig(OUTPUT / "figures" / f"fig{number}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def image_grid(images, rows, columns, title, row_labels=None, column_labels=None):
    """Um único mosaico: eixos identificam imagens e linhas, não intensidade."""
    values = images.detach().cpu().reshape(rows, columns, 28, 28).numpy()
    canvas = np.ones((rows * 30, columns * 30))
    for row in range(rows):
        for col in range(columns):
            canvas[row * 30:row * 30 + 28, col * 30:col * 30 + 28] = values[row, col]
    fig, ax = plt.subplots(figsize=(max(8, columns), max(2.6, rows)))
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    ax.set(title=title, xlabel="Coluna", ylabel="Linha")
    ax.set_xticks(np.arange(columns) * 30 + 13.5,
                  column_labels if column_labels is not None else range(1, columns + 1))
    ax.set_yticks(np.arange(rows) * 30 + 13.5,
                  row_labels if row_labels is not None else range(1, rows + 1))
    fig.tight_layout()
    return fig, ax


@torch.no_grad()
def make_figures(models, results, test, device):
    model16, model2 = models
    model16.eval()
    model2.eval()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    history = results["d16"]["history"]
    for ax, term, name in zip(axes, ("reconstruction", "kl"), ("Reconstrução BCE", "KL")):
        for split, label in (("train", "Treino"), ("validation", "Validação")):
            ax.plot(range(1, EPOCHS + 1), [row[split][term] for row in history], label=label)
        ax.set(title=name, xlabel="Época", ylabel="Nats por imagem", xticks=[1, 5, 10, 15, 20])
        ax.legend()
    fig.suptitle("Figura 1 — Componentes da loss, d = 16")
    fig.tight_layout()
    save_figure(fig, 1)

    indices = [int((test.targets == digit).nonzero()[0].item()) for digit in range(10)]
    images = torch.stack([test[index][0] for index in indices]).to(device)
    reconstruction = model16.decode(model16.encode(images)[0])
    fig, ax = image_grid(torch.cat((images.flatten(1), reconstruction)), 2, 10,
                         "Figura 2 — Originais e reconstruções de mu, d = 16",
                         ["Original", "Reconstrução"], list(range(10)))
    ax.set_xlabel("Dígito (primeira ocorrência no teste)")
    save_figure(fig, 2)

    torch.manual_seed(0)
    generated = model16.decode(torch.randn(64, 16, device=device))
    fig, _ = image_grid(generated, 8, 8, "Figura 3 — 64 amostras do prior, d = 16, seed = 0")
    save_figure(fig, 3)

    means, labels = [], []
    for x, y in DataLoader(test, batch_size=BATCH_SIZE):
        means.append(model2.encode(x.to(device))[0].cpu())
        labels.append(y)
    means, labels = torch.cat(means).numpy(), torch.cat(labels).numpy()
    assert means.shape == (10_000, 2) and np.isfinite(means).all()
    fig, ax = plt.subplots(figsize=(8, 6))
    for digit in range(10):
        points = means[labels == digit]
        ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.5,
                   color=plt.get_cmap("tab10")(digit), label=str(digit))
    ax.set(title="Figura 4 — Médias latentes das 10.000 imagens de teste, d = 2",
           xlabel="mu_1", ylabel="mu_2")
    ax.legend(title="Dígito", markerscale=3, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_figure(fig, 4)

    quantiles = np.linspace(0.05, 0.95, 15)
    coordinates = norm.ppf(quantiles)
    # mu_2 cresce de baixo para cima, na mesma orientação do scatter.
    points = np.array([(a, b) for b in coordinates[::-1] for a in coordinates])
    plane = model2.decode(torch.tensor(points, dtype=torch.float32, device=device))
    fig, ax = image_grid(plane, 15, 15, "Figura 5 — Plano decodificado em quantis normais, d = 2",
                         [f"{v:.2f}" for v in coordinates[::-1]],
                         [f"{v:.2f}" for v in coordinates])
    ax.set(xlabel="z_1 = Phi^-1(q)", ylabel="z_2 = Phi^-1(q)")
    save_figure(fig, 5)

    # Escolha fixa anterior à inspeção: primeiras ocorrências de 0 e 1.
    a, b = indices[0], indices[1]
    endpoints = torch.stack([test[a][0], test[b][0]]).to(device).flatten(1)
    mu = model16.encode(endpoints)[0]
    t = torch.linspace(0, 1, 10, device=device).unsqueeze(1)
    pixels = (1 - t) * endpoints[0] + t * endpoints[1]
    latent = model16.decode((1 - t) * mu[0] + t * mu[1])
    fig, ax = image_grid(torch.cat((pixels, latent)), 2, 10,
                         f"Figura 6 — Interpolação: teste {a} (0) e {b} (1), d = 16",
                         ["Pixels", "Latente"], [f"{v:.2f}" for v in t.flatten().tolist()])
    ax.set_xlabel("t")
    save_figure(fig, 6)
    return {"first_test_indices_by_digit": indices, "interpolation_indices": [a, b],
            "scatter_points": len(means), "plane_quantiles": quantiles.tolist()}


def main():
    # CPU com uma thread evita oversubscription; CUDA é usado se disponível.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checks = validate_implementation()
    train_full = datasets.MNIST(ROOT / "data", train=True, download=True,
                               transform=transforms.ToTensor())
    train, val = random_split(train_full, [50_000, 10_000],
                             generator=torch.Generator().manual_seed(SEED))
    again, _ = random_split(train_full, [50_000, 10_000],
                            generator=torch.Generator().manual_seed(SEED))
    assert train.indices == again.indices
    assert len(set(train.indices) & set(val.indices)) == 0
    assert len(set(train.indices) | set(val.indices)) == 60_000
    assert train_full.data.shape == (60_000, 28, 28)
    assert train_full.data.dtype == torch.uint8
    assert train_full.data.min() == 0 and train_full.data.max() == 255
    batch = next(iter(DataLoader(val, batch_size=BATCH_SIZE)))[0]
    assert batch.shape == (128, 1, 28, 28) and batch.dtype == torch.float32
    assert batch.min() == 0 and batch.max() == 1 and torch.isfinite(batch).all()
    checks.update({"split_reproducible_disjoint_complete": True, "data_range_shape": True})
    results = {"config": {"seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                          "learning_rate": LEARNING_RATE, "beta": 1, "device": str(device),
                          "threads": torch.get_num_threads(), "train": len(train),
                          "validation": len(val), "test": 10_000},
               "versions": {"python": platform.python_version(), "torch": torch.__version__,
                            "torchvision": torchvision.__version__, "numpy": np.__version__,
                            "matplotlib": matplotlib.__version__, "scipy": scipy.__version__},
               "checks": checks}
    models = []
    # Treinar AMBOS antes de carregar/avaliar teste: nenhuma decisão vem do teste.
    for d in (16, 2):
        model, record = train_model(d, train, val, device)
        models.append(model)
        results[f"d{d}"] = record
    test = datasets.MNIST(ROOT / "data", train=False, download=True,
                         transform=transforms.ToTensor())
    assert test.data.shape == (10_000, 28, 28)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE,
                             generator=torch.Generator().manual_seed(SEED))
    for d, model in zip((16, 2), models):
        # Uma amostra de q(z|x) por imagem estima a BCE da ELBO, não decode(mu).
        torch.manual_seed(SEED)
        results[f"d{d}"]["test"] = run_epoch(model, test_loader, device)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    checks["evaluation_no_grad_no_weight_update"] = True
    (OUTPUT / "figures").mkdir(parents=True, exist_ok=True)
    results["figures"] = make_figures(models, results, test, device)
    serialized = json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    (OUTPUT / "results.json").write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
