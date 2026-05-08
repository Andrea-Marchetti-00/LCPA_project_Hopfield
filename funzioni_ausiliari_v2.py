"""
funzioni_ausiliari.py
=====================
Plotting helpers and parallel grid-search for the Hopfield network project.

Public API
----------
plot_corruption_modes       – visualise all corruption strategies side by side
plot_all_results            – original / corrupted / recovered grid
plot_overlap_vs_q           – line plot + shaded std for varying q
plot_overlap_vs_P           – line plot + shaded std for varying P (capacity)
plot_capacity_heatmap       – heatmap avg_overlap(q, P)
plot_mnist_results          – dedicated MNIST recovery display
plot_energy_trace           – energy during recovery iterations
grid_search                 – exhaustive grid search (parallel or sequential)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed


# ──────────────────────────────────────────────────────────────────────────────
# Colour palette (consistent across all plots)
# ──────────────────────────────────────────────────────────────────────────────
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

import torch
import numpy as np

class hopfield_model_torch:
    def __init__(self, patterns, update_method='montecarlo', learning_rule='hebb', R=None, device=None, verbose = "True"):
        """
        Initialize the Hopfield model using PyTorch for GPU acceleration.
        Firts of all we set the device 
        """
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # NB cuda:0 works on my pc, dont know on the others 
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(device)  # stringa tipo "cuda:0" o "cpu"

        """
        Args:
            patterns (list of np.array): List of patterns to store.
            update_method (str): 'synchronous', 'asynchronous' or 'montecarlo'.
            learning_rule (str): 'hebb' or 'local'.
            R (float): Radius for local coupling.
            device (torch.device): Device to run computations on (cuda or cpu).
            verbose (bool): If True, print convergence info.
        """
        self.patterns = patterns 
        self.input_shape = patterns[0].shape
        self.dimension = patterns[0].size
        self.update_method = update_method
        self.learning_rule = learning_rule
        self.R = R
        self.verbose = verbose

        # step 0 Convert patterns to PyTorch tensors one time for all
        self.stored_patterns = [torch.from_numpy(p.flatten()).to(self.device) for p in patterns]
        # 1. crea J di zeri
        self.J = torch.zeros((self.dimension, self.dimension), device=self.device) 
        # 2. chiama train() che SOVRASCRIVE self.J
        self.train()    # i recall in the init file so it s executed everytime we recall the method hopfield_model_torch 
                        #(for instance, if we use m = HopfieldModelTorch(patterns), m.J doesn t exist yet, and all the methods that use self.J would cause an error)
        # d ora in poi tutti i metodi che richiamiamo (dal main si intende) useranno la J che ho definito qua

    # defining the following functions:
    #   train() use to compute J matrix with different methods (Hebb= standard, but also storkey and montecarlo)
    #       the steps are simple: torch zeros create the matrix with the right dimensions, 
    #       then we compute pattern*pattern and sum the result on the J matrix (iterate over this process with += in the for cycle)
    #       in the end we divede over the dimension of the matrix (=number of neurons = self.dimension) an ddecorrelate all the patterns by filling the diagonal with zeros
    def train(self):
        patterns = self.stored_patterns

        if self.learning_rule == 'hebb':
           for p in patterns:
               self.J += torch.outer(p, p)
           self.J /= self.dimension
           self.J.fill_diagonal_(0) # To avoid self-correlations, otherwise each pattern would naturally tend to have high correlation with itself
        
        elif self.learning_rule == 'local':
            if self.R is None:
                raise ValueError("For 'local', R must be specified.")
            if len(self.input_shape) != 2:
                raise ValueError("'local' is only implemented for 2D patterns.")
            height, width = self.input_shape
            positions = torch.tensor([(i, j) for i in range(height) for j in range(width)], device=self.device)
            dists = torch.sqrt(torch.sum((positions[:, None, :] - positions[None, :, :])**2, dim=2))
            local_mask = dists <= self.R
            for p in patterns:
                self.J += torch.outer(p, p) * local_mask
            self.J /= self.dimension
            self.J.fill_diagonal_(0)

        elif self.learning_rule == 'storkey':
            for xi in patterns:
                h = torch.matmul(self.J, xi)
                delta = (torch.outer(xi, xi) - torch.outer(xi, h) - torch.outer(h, xi)) / self.dimension
                self.J += delta
                self.J.fill_diagonal_(0)
        else:
            raise ValueError("Invalid learning rule.")


    # defining the function used to update the state in order to find new pattern at each iteration
    # takes an initial state and updates it with the specified method (synch., async., montec.)
    # sync. meth.: all neurons updated together (torch.sign defines the new state for each neuron)
    #       if local field is zero the neuron stays in the same state (otherwise it changes the value)
    #       iterate "steps" times
    # async. meth.: neurons are updated one by one
    #       number of updates = self.dimension
    #       takes a random i and calculates its local field with h = torch.matmul...
    #       then updates only that neuron with pattern[i]=toch.sign...
    # montec. method: stochastic update
    #       if the flip lower the energy we always accept it, otherwise we cccept it qith a prob that depernd from the taemperature and the typo of schedule
    #       there are 3 types of schedule: classic, exponential and logarithmic 
    def update(self, state, steps=1, temperature=0.0, alpha=0.0, global_iter=None, schedule='classic'):
        # pattern = state.flatten() # non garantisce che sia un tensor torch
        pattern = torch.tensor(state.flatten(), dtype=torch.float32, device=self.device)
        if self.update_method == 'synchronous':
            for _ in range(steps):
                new_pattern = torch.sign(torch.matmul(self.J, pattern))
                new_pattern[new_pattern == 0] = pattern[new_pattern == 0]
                pattern = new_pattern
            return pattern
        elif self.update_method == 'asynchronous':
            for _ in range(steps):
                for _ in range(self.dimension):
                    # i = np.random.randint(0, self.dimension)
                    i = torch.randint(0, self.dimension, (1,)).item()
                    h = torch.matmul(self.J[i, :], pattern)
                    pattern[i] = torch.sign(h) if h != 0 else pattern[i]
            return pattern
        elif self.update_method == 'montecarlo':
            for t in range(self.dimension):
                # calcola T in base a global_iter o t
                iter_idx = global_iter if global_iter is not None else t
                if schedule == 'classic':
                    T = temperature / (1 + alpha * iter_idx)
                elif schedule == 'exponential':
                    T = temperature * np.exp(-alpha * iter_idx)
                elif schedule == 'logarithmic':
                    T = temperature / (1 + alpha * np.log(1 + iter_idx))
                else:
                    raise ValueError("Schedule not recognized.")

                i = torch.randint(0, self.dimension, (1,)).item()
                h = torch.dot(self.J[i, :], pattern)
                delta_E = 2 * pattern[i] * h
                if delta_E <= 0:
                    pattern[i] = -pattern[i]
                elif T > 0 and torch.rand(1).item() < np.exp(-delta_E.item() / T):
                    pattern[i] = -pattern[i]
            return pattern
        else:
            raise ValueError("Invalid update_method.")

    # we can compute the energy of the state in order to understand if it's stable or not (minumum are attractors)  
    def energy(self, state):
        s = torch.tensor(state.flatten(), dtype=torch.float32, device=self.device)
        return -0.5 * torch.dot(s, torch.matmul(self.J, s)) 
    
    # Restituisce una stima teorica della capacità massima della rete di Hopfield classica 
    def storage_limit(self):
        return 0.138 * self.dimension
    
    # we compute how much overlap there is between the original and the recoverd pattern
    def overlap(self, pattern1, pattern2, absolute=False):
        p1 = torch.tensor(pattern1.flatten(), dtype=torch.float32, device=self.device)
        p2 = torch.tensor(pattern2.flatten(), dtype=torch.float32, device=self.device)
        val = torch.dot(p1, p2) / self.dimension
        return val.abs() if absolute else val

    # mesure how much the memorized patterns interfere among them 
    # (compute the overlap for all the pairs of patterns stored and sums the absolute values, then computes the mean)
    def memory_interference(self):
        P = len(self.stored_patterns)
        if P < 2:
            return 0
        total = 0
        count = 0
        for mu in range(P):
            for nu in range(mu+1, P):
                ov = torch.dot(self.stored_patterns[mu], self.stored_patterns[nu]) / self.dimension
                total += ov.abs()
                count += 1
        return total  / count  
    
    # we try to recover the original image with the following steps:
    #       1) takes corrupted pattern
    #       2) updates it with the method update defined above
    #       3) check if something has changed con torch.equal()
    #       4) (if it doesnt for many iterations (checks> convergence_check), it stops)
    def correct(self, corrupted_pattern, max_iter=100, convergence_check=1, temperature=0.0, alpha=0.0, schedule='classic'):
        pattern = torch.tensor(corrupted_pattern.flatten(), dtype=torch.float32, device=self.device)   
        check = 0
        for i in range(max_iter):
            new_pattern = self.update(pattern, steps=1, temperature=temperature, alpha=alpha, global_iter=i, schedule=schedule)
            if torch.equal(pattern, new_pattern):
                check += 1
                if check >= convergence_check:
                    break
            else:
                check = 0
            pattern = new_pattern
        if self.verbose:
            print(f"Converged after {i+1} iterations")
        return pattern.cpu().numpy() # .cpu() to 
    
    # versione batch di quanto visto con correct, ma di fatto fa la stessa cosa
    # batch version for correct function, it does the same thing
    def correct_patterns(self, corrupted_patterns, max_iter=100, convergence_check=1, temperature=0.0, alpha=0.0, schedule='classic'):
        corrected_patterns = np.zeros_like(corrupted_patterns)
        n = corrupted_patterns.shape[0]
        for i in range(n):
            corrected = self.correct(corrupted_patterns[i], max_iter, convergence_check, temperature, alpha, schedule)
            corrected_patterns[i] = corrected.reshape(self.input_shape)
        return corrected_patterns
    


# ──────────────────────────────────────────────────────────────────────────────
# 1.  CORRUPTION VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_corruption_modes(pattern, q=0.7, figsize=None, filepath=None):
    """
    Show a single pattern corrupted with all four available methods.

    Each column: Original | corrupted | local retention-probability map.
    """
    from patterns_v2 import corrupt_patterns, corruption_map

    methods = ['random', 'radial', 'block', 'gradient']
    labels  = ['Random\n(uniform)', 'Radial\n(centre preserved)',
               'Block\n(patch removed)', 'Gradient\n(top → bottom)']
    kwargs  = [{}, {}, {'block_fraction': 0.5},
               {'direction': 'vertical', 'q_min': max(0, q - 0.6)}]

    n = len(methods)
    fig_w = figsize[0] if figsize else 3 * n
    fig_h = figsize[1] if figsize else 6
    fig, axes = plt.subplots(3, n, figsize=(fig_w, fig_h))

    for col, (m, lbl, kw) in enumerate(zip(methods, labels, kwargs)):
        corrupted = corrupt_patterns([pattern], q, method=m, **kw)[0]
        q_map     = corruption_map(pattern.shape, q, method=m, **kw)

        # Row 0: original
        ax = axes[0, col]
        ax.imshow((pattern + 1) / 2, cmap='gray', vmin=0, vmax=1)
        ax.set_title(lbl, fontsize=9)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel('Original', fontsize=9)

        # Row 1: corrupted
        ax = axes[1, col]
        ax.imshow((corrupted + 1) / 2, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel('Corrupted', fontsize=9)

        # Row 2: retention-probability map
        ax = axes[2, col]
        im = ax.imshow(q_map, cmap='RdYlGn', vmin=0, vmax=1)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel('Retention prob.', fontsize=9)

    fig.colorbar(im, ax=axes[2, :], shrink=0.6, label='Retention probability q')
    fig.suptitle(f'Corruption methods  (global q = {q})', fontsize=12,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 2.  ORIGINAL / CORRUPTED / RECOVERED  GRID
# ──────────────────────────────────────────────────────────────────────────────

def plot_all_results(patterns, corrupted, recovered,
                     model=None, figsize=None, filepath=None,
                     labels=None):
    """
    Three-row grid: original | corrupted | recovered.

    model : hopfield_model_torch instance — if provided, the overlap between
            the recovered and the original is shown as a subtitle.
    labels: list of column labels (e.g. digit names)
    """
    P = len(patterns)
    fw = figsize[0] if figsize else max(8, 2 * P)
    fh = figsize[1] if figsize else 5
    fig, axes = plt.subplots(3, P, figsize=(fw, fh))
    if P == 1:
        axes = axes[:, None]

    row_labels = ['Original', 'Corrupted', 'Recovered']
    for row, (row_ax, rlab) in enumerate(zip(axes, row_labels)):
        for col, ax in enumerate(row_ax):
            if row == 0:
                img = patterns[col]
            elif row == 1:
                img = corrupted[col]
            else:
                img = recovered[col]

            ax.imshow(img.squeeze().reshape(patterns[col].shape),
                      cmap='gray', vmin=-1, vmax=1)
            ax.axis('off')

            if row == 0:
                title = labels[col] if labels else f'Pattern {col}'
                ax.set_title(title, fontsize=9)

            # if col == 0:
            #     ax.set_ylabel(rlab, fontsize=9)
            if col == 0:
                ax.annotate(rlab, xy=(0, 0.5), xytext=(-0.15, 0.5),
                            xycoords='axes fraction', textcoords='axes fraction',
                            fontsize=9, ha='right', va='center', rotation=90)

            if row == 2 and model is not None:
                ov = model.overlap(patterns[col], recovered[col])
                ax.set_xlabel(f'ov = {float(ov):.2f}', fontsize=8)

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 3.  MNIST  DEDICATED DISPLAY
# ──────────────────────────────────────────────────────────────────────────────

def plot_mnist_results(patterns, corrupted, recovered, model=None,
                       filepath=None, figsize=None):
    """Like plot_all_results but with digit labels 0-9."""
    labels = [str(d) for d in range(len(patterns))]
    plot_all_results(patterns, corrupted, recovered,
                     model=model, figsize=figsize,
                     filepath=filepath, labels=labels)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  ENERGY TRACE
# ──────────────────────────────────────────────────────────────────────────────

# def plot_energy_trace(model, corrupted_pattern, max_iter=100,
#                       temperature=0.0, alpha=0.0, schedule='classic',
#                       filepath=None, figsize=(7, 3.5)):
#     """
#     Run the recovery step by step and record the energy at each iteration.
#     Returns the recovered pattern and the energy trace.
#     """
#     import torch

#     pattern = torch.tensor(corrupted_pattern.flatten(),
#                            dtype=torch.float32, device=model.device)
#     energies = [float(model.energy(pattern.cpu().numpy()))]

#     for i in range(max_iter):
#         pattern = model.update(pattern.cpu().numpy(), steps=1,
#                                temperature=temperature, alpha=alpha,
#                                global_iter=i, schedule=schedule)
#         pattern = torch.tensor(pattern.flatten(),
#                                dtype=torch.float32, device=model.device)
#         energies.append(float(model.energy(pattern.cpu().numpy())))

#     fig, ax = plt.subplots(figsize=figsize)
#     ax.plot(energies, color=PALETTE[0], linewidth=1.8)
#     ax.set_xlabel('Iteration', fontsize=12)
#     ax.set_ylabel('Energy  $E = -\\frac{1}{2}\\mathbf{S}^T J \\mathbf{S}$', fontsize=12)
#     ax.set_title('Energy during recovery', fontsize=13, fontweight='bold')
#     ax.grid(True, linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     if filepath:
#         plt.savefig(filepath, bbox_inches='tight', dpi=150)
#     plt.show()

#     return pattern.cpu().numpy().reshape(corrupted_pattern.shape), energies


def plot_energy_trace(model, corrupted_patterns, max_iter=100,
                      temperature=0.0, alpha=0.0, schedule='classic',
                      filepath=None, figsize=(7, 3.5)):
    import torch

    all_energies = []

    for corrupted_pattern in corrupted_patterns:
        pattern = torch.tensor(corrupted_pattern.flatten(),
                               dtype=torch.float32, device=model.device)
        energies = [float(model.energy(pattern.cpu().numpy()))]

        for i in range(max_iter):
            pattern = model.update(pattern.cpu().numpy(), steps=1,
                                   temperature=temperature, alpha=alpha,
                                   global_iter=i, schedule=schedule)
            pattern = torch.tensor(pattern.flatten(),
                                   dtype=torch.float32, device=model.device)
            energies.append(float(model.energy(pattern.cpu().numpy())))

        all_energies.append(energies)

    all_energies = np.array(all_energies)
    mean_energy  = all_energies.mean(axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    for trace in all_energies:
        ax.plot(trace, color=PALETTE[0], linewidth=1.2, alpha=0.3)
    ax.plot(mean_energy, color='crimson', linewidth=2.0, label='Mean')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Energy  $E = -\\frac{1}{2}\\mathbf{S}^T J \\mathbf{S}$', fontsize=12)
    ax.set_title('Energy during recovery', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.show()

    return all_energies


# ──────────────────────────────────────────────────────────────────────────────
# 5.  OVERLAP vs Q
# ──────────────────────────────────────────────────────────────────────────────

def plot_overlap_vs_q(results_dict, n_patterns_list, N, filepath=None,
                      figsize=(8, 4.5)):
    """
    Plot average overlap vs q for several values of P (number of patterns).

    results_dict : { P: { q: (avg_overlap, std_overlap) } }
    """
    fig, ax = plt.subplots(figsize=figsize)

    for i, P in enumerate(n_patterns_list):
        color = PALETTE[i % len(PALETTE)]
        data  = results_dict[P]
        qs    = sorted(data.keys())
        avgs  = np.array([data[q][0] for q in qs])
        stds  = np.array([data[q][1] for q in qs])

        ax.plot(qs, avgs, marker='o', markersize=5, linewidth=2,
                color=color, label=f'P = {P}')
        ax.fill_between(qs,
                        np.clip(avgs - stds, -1, 1),
                        np.clip(avgs + stds, -1, 1),
                        color=color, alpha=0.15)

    ax.axhline(0, color='grey', linestyle=':', linewidth=1)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1.2,
               label='q = 0.5 (50 % noise)')
    ax.set_xlabel('Retention probability q', fontsize=12)
    ax.set_ylabel('Average overlap  $m$', fontsize=12)
    ax.set_title(f'Overlap vs corruption level  (N = {N})',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1); ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 6.  OVERLAP vs P  (capacity)
# ──────────────────────────────────────────────────────────────────────────────

def plot_overlap_vs_P(results_dict, q_list, N, filepath=None, figsize=(8, 4.5)):
    """
    Plot average overlap vs P for several values of q, with theoretical limit.

    results_dict : { q: { P: (avg_overlap, std_overlap) } }
    """
    fig, ax = plt.subplots(figsize=figsize)
    theoretical_limit = 0.138 * N

    for i, q in enumerate(q_list):
        color = PALETTE[i % len(PALETTE)]
        data  = results_dict[q]
        Ps    = sorted(data.keys())
        avgs  = np.array([data[P][0] for P in Ps])
        stds  = np.array([data[P][1] for P in Ps])

        ax.plot(Ps, avgs, marker='o', markersize=5, linewidth=2,
                color=color, label=f'q = {q:.2f}')
        ax.fill_between(Ps,
                        np.clip(avgs - stds, -1, 1),
                        np.clip(avgs + stds, -1, 1),
                        color=color, alpha=0.15)

    ax.axvline(theoretical_limit, color='crimson', linestyle='--', linewidth=2,
               label=f'Theoretical limit  $P_c ≈ 0.138N$ = {theoretical_limit:.0f}')
    ax.set_xlabel('Number of stored patterns  P', fontsize=12)
    ax.set_ylabel('Average overlap  $m$', fontsize=12)
    ax.set_title(f'Storage capacity  (N = {N})',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 7.  CAPACITY HEATMAP
# ──────────────────────────────────────────────────────────────────────────────

def plot_capacity_heatmap(results_dict, q_list, P_list, N, filepath=None,
                          figsize=(9, 5)):
    """
    Heatmap of avg_overlap on axes (q, P).

    results_dict : { q: { P: (avg_overlap, std_overlap) } }
    """
    import seaborn as sns

    data = np.array([[results_dict[q][P][0] for P in P_list] for q in q_list])
    fig, ax = plt.subplots(figsize=figsize)

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    sns.heatmap(data, xticklabels=P_list,
                yticklabels=[f'{q:.2f}' for q in q_list],
                cmap='RdYlGn', norm=norm,
                linewidths=0.3, linecolor='white',
                cbar_kws={'label': 'Average overlap $m$'},
                ax=ax)

    # mark theoretical limit
    for j, P in enumerate(P_list):
        if P > 0.138 * N:
            ax.axvline(j, color='royalblue', linewidth=1.5,
                       linestyle='--')
            ax.text(j + 0.1, 0.3, f'$P_c≈{0.138*N:.0f}$',
                    color='royalblue', fontsize=8, va='top')
            break

    ax.set_xlabel('Number of stored patterns  P', fontsize=12)
    ax.set_ylabel('Retention probability  q', fontsize=12)
    ax.set_title(f'Average overlap heatmap  (N = {N})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 8.  GRID SEARCH
# ──────────────────────────────────────────────────────────────────────────────

def grid_search(param_grid, generate_fn, corrupt_fn, model_class,
                n_repeats=3, max_iter=100, seed=42,
                corruption_method='random', verbose=True):
    """
    Exhaustive grid search over Hopfield hyper-parameters on random patterns.

    Parameters
    ----------
    param_grid   : dict with keys (lists or scalars):
                     shape, n_patterns, q, update_method, learning_rule,
                     R, temperature, alpha, schedule
    generate_fn  : callable  – e.g. generate_random_patterns(shape, P, seed=...)
    corrupt_fn   : callable  – e.g. corrupt_patterns(patterns, q, method=...)
    model_class  : class     – e.g. hopfield_model_torch
    n_repeats    : int  – independent repetitions per configuration
    max_iter     : int  – max recovery iterations
    seed         : int  – base random seed
    corruption_method : str  – passed to corrupt_fn as method=
    verbose      : bool

    Returns
    -------
    list of result dicts, sorted by avg_overlap descending
    """
    def _listify(v):
        return v if isinstance(v, (list, np.ndarray)) else [v]

    keys = ['shape', 'n_patterns', 'q', 'update_method', 'learning_rule',
            'R', 'temperature', 'alpha', 'schedule']
    combos = list(product(*[_listify(param_grid.get(k, [None])) for k in keys]))

    if verbose:
        print(f"Grid search: {len(combos)} configs × {n_repeats} repeats "
              f"= {len(combos) * n_repeats} evaluations")

    results = []
    for i, (shape, n_pat, q, upd, lr, R, T, al, sched) in enumerate(combos):
        overlaps = []
        for rep in range(n_repeats):
            rep_seed = (seed + rep) if seed is not None else None
            patterns  = generate_fn(shape, n_pat, seed=rep_seed)
            corrupted = corrupt_fn(patterns, q, method=corruption_method,
                                   seed=rep_seed)
            model = model_class(patterns, update_method=upd,
                                learning_rule=lr, R=R, verbose=False)
            recovered = model.correct_patterns(
                np.array([c.reshape(shape) for c in corrupted]),
                max_iter=max_iter,
                temperature=T   if T  is not None else 0.0,
                alpha=al        if al is not None else 0.0,
                schedule=sched  if sched is not None else 'classic')
            for orig, rec in zip(patterns, recovered):
                overlaps.append(float(model.overlap(orig, rec, absolute=False)))

        results.append({
            'shape': shape, 'n_patterns': n_pat, 'q': q,
            'update_method': upd, 'learning_rule': lr, 'R': R,
            'temperature': T, 'alpha': al, 'schedule': sched,
            'avg_overlap': float(np.mean(overlaps)),
            'std_overlap': float(np.std(overlaps)),
        })
        if verbose and (i + 1) % max(1, len(combos) // 10) == 0:
            print(f"  {i+1}/{len(combos)} done …")

    results.sort(key=lambda r: r['avg_overlap'], reverse=True)

    if verbose:
        best = results[0]
        print("\nBest configuration:")
        for k, v in best.items():
            print(f"  {k:20s}: {v}")

    return results
