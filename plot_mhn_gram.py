"""
plot_mhn_gram.py
=================
For each MNIST digit (0-9), build the **Modern Hopfield Network** pattern
matrix X and analyse its structure.

The MHN has no coupling matrix J.  The equivalent object is the
pattern-correlation (Gram) matrix

    G = (1/N) X^T X      (size P × P)

whose entries are the pairwise overlaps between stored patterns of that digit.
We plot G and its eigenvalue spectrum for each digit.

Usage:  python plot_mhn_gram.py
"""

import os, sys, numpy as np, matplotlib.pyplot as plt, torch
from pathlib import Path

_project_dir = Path(__file__).resolve().parent
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs('plots', exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Load MNIST at 14×14 (same as classical case for fair comparison)
# ══════════════════════════════════════════════════════════════════════════════
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((14, 14)),
    transforms.ToTensor(),
])
train_dataset = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)

per_digit_images = {d: [] for d in range(10)}
for img_t, label in train_dataset:
    img_np = img_t.squeeze(0).numpy()
    img_bin = np.where(img_np > 0.5, 1.0, -1.0).astype(np.float32)
    per_digit_images[label].append(img_bin)

N_PATTERNS_PER_DIGIT = 100   # same order as classical case

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Build the pattern matrix X for each digit  (size N × P)
# ══════════════════════════════════════════════════════════════════════════════
digit_X = {}   # X[d] is (N, P)
for d in range(10):
    idx = np.random.choice(len(per_digit_images[d]),
                           size=N_PATTERNS_PER_DIGIT, replace=False)
    patterns_d = [per_digit_images[d][i] for i in idx]
    # stack into (N, P)
    X = np.column_stack([p.ravel() for p in patterns_d]).astype(np.float32)
    digit_X[d] = X

N = digit_X[0].shape[0]

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Gram matrix  G = (1/N) X^T X   — the MHN analog of J
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle(
    'MHN — Pattern Gram matrix  $G = \\frac{1}{N}X^\\top X$  per digit\n'
    f'(N={N}, P={N_PATTERNS_PER_DIGIT})',
    fontsize=14, fontweight='bold', y=1.02
)

for d, ax in enumerate(axes.flat):
    X = digit_X[d]
    G = (X.T @ X) / N
    vmax = abs(G).max()

    im = ax.imshow(G, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   interpolation='nearest', aspect='equal')
    ax.set_title(f'Digit {d}', fontsize=12, fontweight='bold')
    ax.set_xlabel('pattern $\\nu$', fontsize=8)
    ax.set_ylabel('pattern $\\mu$', fontsize=8)
    ax.tick_params(axis='both', which='both', length=0, labelsize=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('plots/MHN_gram_per_digit.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved: plots/MHN_gram_per_digit.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Eigenvalue spectrum of G
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 5, figsize=(20, 8))
fig2.suptitle(
    'MHN — Eigenvalue spectrum of the Gram matrix $G$ per digit\n'
    f'(N={N}, P={N_PATTERNS_PER_DIGIT})',
    fontsize=14, fontweight='bold', y=1.02
)

for d, ax in enumerate(axes2.flat):
    X = digit_X[d]
    G = (X.T @ X) / N
    eigvals = np.linalg.eigvalsh(G)
    ax.hist(eigvals, bins=30, color='steelblue', edgecolor='white', linewidth=0.4)
    ax.set_yscale('log')
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    ax.set_title(f'Digit {d}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Eigenvalue', fontsize=8)
    if d % 5 == 0:
        ax.set_ylabel('Count (log)', fontsize=8)

plt.tight_layout()
plt.savefig('plots/MHN_eigenspectra_per_digit.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved: plots/MHN_eigenspectra_per_digit.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Overlay of all spectra
# ══════════════════════════════════════════════════════════════════════════════
PALETTE = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
           "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]

fig3, ax3 = plt.subplots(figsize=(10, 6))
for d in range(10):
    X = digit_X[d]
    G = (X.T @ X) / N
    eigvals = np.linalg.eigvalsh(G)
    counts, bins = np.histogram(eigvals, bins=60, density=True)
    bc = (bins[:-1] + bins[1:]) / 2
    ax3.plot(bc, counts, color=PALETTE[d], lw=1.8, label=f'Digit {d}')

ax3.set_yscale('log')
ax3.axvline(0, color='grey', linestyle='--', linewidth=1)
ax3.set_xlabel('Eigenvalue', fontsize=12)
ax3.set_ylabel('Density (log)', fontsize=12)
ax3.set_title(f'MHN — Gram matrix eigenvalue spectra, all digits overlaid  (N={N}, P={N_PATTERNS_PER_DIGIT})',
              fontsize=13, fontweight='bold')
ax3.legend(fontsize=9, ncol=2)
ax3.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/MHN_eigenspectra_overlay.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved: plots/MHN_eigenspectra_overlay.png")
