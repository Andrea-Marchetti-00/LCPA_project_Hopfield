"""
plot_j_matrices.py
===================
For each MNIST digit (0-9), build a **classical** Hopfield model using Hebbian
learning trained ONLY on training images of that digit, and plot the coupling
matrix J_{ij} side by side.

Usage:  python plot_j_matrices.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# ── add project root to sys.path ──────────────────────────────────────────────
_project_dir = Path(__file__).resolve().parent
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from funzioni_ausiliari_v2 import hopfield_model_torch

# reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs('plots', exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Load MNIST at 14×14  (so J is 196×196 — manageable to visualise)
# ══════════════════════════════════════════════════════════════════════════════
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((14, 14)),
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)

# organise by digit
per_digit_images = {d: [] for d in range(10)}
for img_t, label in train_dataset:
    img_np = img_t.squeeze(0).numpy()          # (14, 14) in [0, 1]
    img_bin = np.where(img_np > 0.5, 1.0, -1.0).astype(np.float32)
    per_digit_images[label].append(img_bin)

N_PATTERNS_PER_DIGIT = 300   # how many training images per digit

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Build one classical Hopfield model per digit  (Hebbian rule)
# ══════════════════════════════════════════════════════════════════════════════
digit_models = {}
for d in range(10):
    # select N_PATTERNS_PER_DIGIT random images of this digit
    idx = np.random.choice(len(per_digit_images[d]),
                           size=N_PATTERNS_PER_DIGIT, replace=False)
    patterns_d = [per_digit_images[d][i] for i in idx]

    model = hopfield_model_torch(
        patterns_d,
        update_method='synchronous',
        learning_rule='hebb',
        verbose=False,
    )
    digit_models[d] = model

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Plot all 10 J matrices
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle(
    'Coupling matrix $J_{ij}$  —  Classical Hopfield model trained per digit\n'
    f'(N=196, P={N_PATTERNS_PER_DIGIT}, Hebbian rule)',
    fontsize=14, fontweight='bold', y=1.02
)

for d, ax in enumerate(axes.flat):
    J = digit_models[d].J.cpu().numpy()
    vmax = abs(J).max()

    im = ax.imshow(J, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   interpolation='nearest', aspect='equal')
    ax.set_title(f'Digit {d}', fontsize=12, fontweight='bold')
    ax.set_xlabel('neuron $j$', fontsize=8)
    ax.set_ylabel('neuron $i$', fontsize=8)
    ax.tick_params(axis='both', which='both', length=0, labelsize=0)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('plots/J_matrices_per_digit.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved: plots/J_matrices_per_digit.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Also show J structure as zoomed thumbnails (small crops of J)
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 5, figsize=(20, 8))
fig2.suptitle(
    'Zoom into $J_{ij}$  —  first 50 × 50 entries',
    fontsize=14, fontweight='bold', y=1.02
)

for d, ax in enumerate(axes2.flat):
    J = digit_models[d].J.cpu().numpy()[:50, :50]
    vmax = abs(J).max()

    im = ax.imshow(J, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   interpolation='nearest', aspect='equal')
    ax.set_title(f'Digit {d}', fontsize=12, fontweight='bold')
    ax.set_xlabel('neuron $j$', fontsize=8)
    ax.set_ylabel('neuron $i$', fontsize=8)
    ax.tick_params(axis='both', which='both', length=0, labelsize=0)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('plots/J_matrices_per_digit_zoom.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved: plots/J_matrices_per_digit_zoom.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Eigenvalue spectrum of J for each digit
# ══════════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 5, figsize=(20, 8))
fig3.suptitle(
    'Eigenvalue spectrum of $J_{ij}$  —  Classical Hopfield model per digit\n'
    f'(N=196, P={N_PATTERNS_PER_DIGIT}, Hebbian rule)',
    fontsize=14, fontweight='bold', y=1.02
)

for d, ax in enumerate(axes3.flat):
    eigvals = torch.linalg.eigvalsh(digit_models[d].J).cpu().numpy()
    ax.hist(eigvals, bins=30, color='steelblue', edgecolor='white', linewidth=0.4)
    ax.set_yscale('log')
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    ax.set_title(f'Digit {d}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Eigenvalue', fontsize=8)
    if d % 5 == 0:
        ax.set_ylabel('Count (log)', fontsize=8)

plt.tight_layout()
plt.savefig('plots/J_eigenspectra_per_digit.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved: plots/J_eigenspectra_per_digit.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  Combined eigenvalue spectra on one axes for comparison
# ══════════════════════════════════════════════════════════════════════════════
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
           "#bcbd22", "#17becf"]

fig4, ax4 = plt.subplots(figsize=(10, 6))
for d in range(10):
    eigvals = torch.linalg.eigvalsh(digit_models[d].J).cpu().numpy()
    # kernel density estimate via histogram
    counts, bins = np.histogram(eigvals, bins=60, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax4.plot(bin_centers, counts, color=PALETTE[d], lw=1.8, label=f'Digit {d}')

ax4.set_yscale('log')
ax4.axvline(0, color='grey', linestyle='--', linewidth=1)
ax4.set_xlabel('Eigenvalue', fontsize=12)
ax4.set_ylabel('Density (log)', fontsize=12)
ax4.set_title(f'Eigenvalue spectra — all digits overlaid  (N=196, P={N_PATTERNS_PER_DIGIT})',
              fontsize=13, fontweight='bold')
ax4.legend(fontsize=9, ncol=2)
ax4.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/J_eigenspectra_overlay.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved: plots/J_eigenspectra_overlay.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7.  Summary table: eigenvalues + interference
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'Digit':>5} {'P':>3} {'N':>4} {'P_c':>6} {'Interference':>14}")
print("-" * 50)
for d in range(10):
    m = digit_models[d]
    print(f"{d:>5} {N_PATTERNS_PER_DIGIT:>3} {m.dimension:>4} "
          f"{m.storage_limit():>6.1f} {float(m.memory_interference()):>14.4f}")
