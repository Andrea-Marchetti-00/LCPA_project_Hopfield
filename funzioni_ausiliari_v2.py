"""
Hopfield model and plotting helpers for the Hopfield network project.

hopfield_model_torch  - PyTorch Hopfield network (train / update / recovery)
plot_corruption_modes - visualise all corruption strategies side by side
plot_all_results      - original / corrupted / recovered grid
plot_overlap_vs_q     - line plot + shaded std for varying q
plot_overlap_vs_P     - line plot + shaded std for varying P (capacity)
plot_capacity_heatmap - heatmap avg_overlap(q, P)
plot_energy_trace     - energy during recovery iterations
compute_phase_diagram - sample the equilibrium overlap on a (T, alpha) grid
plot_phase_diagram    - plot the (T, alpha) phase diagram
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from patterns_v2 import generate_random_patterns

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


class hopfield_model_torch:
    def __init__(self, patterns, update_method='montecarlo', learning_rule='hebb', R=None, device=None, verbose="True"):

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # NB cuda:0 works on my pc, dont know on the others
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(device)

        self.patterns = patterns
        self.input_shape = patterns[0].shape
        self.dimension = patterns[0].size
        self.update_method = update_method
        self.learning_rule = learning_rule
        self.R = R
        self.verbose = verbose

        # convert patterns to tensors once
        self.stored_patterns = [torch.from_numpy(p.flatten()).to(self.device) for p in patterns]
        self.J = torch.zeros((self.dimension, self.dimension), device=self.device)
        # train() overwrites self.J; we call it here so J exists right after
        # construction and every method that uses self.J works
        self.train()

    def train(self):
        # Build J: sum the outer products pattern*pattern, divide by the number
        # of neurons, and zero the diagonal to remove self-correlations.
        patterns = self.stored_patterns
        self.J.zero_()

        if self.learning_rule == 'hebb':
            for p in patterns:
                self.J += torch.outer(p, p)
            self.J /= self.dimension
            self.J.fill_diagonal_(0)

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

        else:
            raise ValueError("Invalid learning rule.")

    def update(self, state, steps=1, temperature=0.0, alpha=0.0, global_iter=None, schedule='classic'):
        # Update the state with the chosen method. Synchronous updates all
        # neurons at once; asynchronous updates them one by one in random order;
        # montecarlo is a stochastic Metropolis update with a temperature
        # schedule ('classic', 'exponential' or 'logarithmic').
        pattern = torch.as_tensor(state, dtype=torch.float32, device=self.device).flatten()

        if self.update_method == 'synchronous':
            for _ in range(steps):
                new_pattern = torch.sign(torch.matmul(self.J, pattern))
                new_pattern[new_pattern == 0] = pattern[new_pattern == 0]
                pattern = new_pattern
            return pattern
        elif self.update_method == 'asynchronous':
            for _ in range(steps):
                for _ in range(self.dimension):
                    i = torch.randint(0, self.dimension, (1,)).item()
                    h = torch.matmul(self.J[i, :], pattern)
                    pattern[i] = torch.sign(h) if h != 0 else pattern[i]
            return pattern
        elif self.update_method == 'montecarlo':
            for t in range(self.dimension):
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

    def energy(self, state):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).flatten()
        return -0.5 * torch.dot(s, torch.matmul(self.J, s))

    def storage_limit(self):
        # theoretical capacity of the classical Hopfield network
        return 0.138 * self.dimension

    def overlap(self, pattern1, pattern2, absolute=False):
        p1 = torch.as_tensor(pattern1, dtype=torch.float32, device=self.device).flatten()
        p2 = torch.as_tensor(pattern2, dtype=torch.float32, device=self.device).flatten()
        val = torch.dot(p1, p2) / self.dimension
        return val.abs() if absolute else val

    def memory_interference(self):
        # mean absolute overlap over all pairs of stored patterns
        P = len(self.stored_patterns)
        if P < 2:
            return 0
        total = 0
        count = 0
        for mu in range(P):
            for nu in range(mu + 1, P):
                ov = torch.dot(self.stored_patterns[mu], self.stored_patterns[nu]) / self.dimension
                total += ov.abs()
                count += 1
        return total / count

    def correct(self, corrupted_pattern, max_iter=100, convergence_check=1, temperature=0.0, alpha=0.0, schedule='classic'):
        # Recover the original pattern: repeatedly update the corrupted pattern
        # and stop once it stops changing for `convergence_check` iterations.
        pattern = torch.as_tensor(corrupted_pattern, dtype=torch.float32, device=self.device).flatten()
        check = 0
        # for asynchronous, one step already touches ~63% of the spins, so we
        # require a stricter convergence check
        if self.update_method == 'asynchronous' and convergence_check < 5:
            convergence_check = 5
        for i in range(max_iter):
            new_pattern = self.update(pattern.clone(), steps=1, temperature=temperature, alpha=alpha, global_iter=i, schedule=schedule)
            if torch.equal(pattern, new_pattern):
                check += 1
                if check >= convergence_check:
                    break
            else:
                check = 0
            pattern = new_pattern
        if self.verbose:
            print(f"Converged after {i+1} iterations")
        return pattern.cpu().numpy()

    def correct_patterns(self, corrupted_patterns, max_iter=100, convergence_check=1, temperature=0.0, alpha=0.0, schedule='classic'):
        # batch version of correct()
        corrected_patterns = np.zeros_like(corrupted_patterns)
        n = corrupted_patterns.shape[0]
        for i in range(n):
            corrected = self.correct(corrupted_patterns[i], max_iter, convergence_check, temperature, alpha, schedule)
            corrected_patterns[i] = corrected.reshape(self.input_shape)
        return corrected_patterns


def pixel_diff_map(pattern_a, pattern_b):

    a = np.asarray(pattern_a)
    b = np.asarray(pattern_b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return (a != b).astype(np.float32)


def plot_pixel_diff(pattern_a, pattern_b, ax=None,
                    color='#d62728', label_a='A', label_b='B',
                    show_count=True, figsize=(3, 3)):

    from matplotlib.colors import ListedColormap

    diff = pixel_diff_map(pattern_a, pattern_b)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    diff_cmap = ListedColormap(['white', color])
    ax.imshow(diff, cmap=diff_cmap, vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])

    if show_count:
        n_diff, n_tot = int(diff.sum()), diff.size
        ax.set_xlabel(f'{n_diff}/{n_tot} differ ({100*n_diff/n_tot:.0f}%)\n'
                      f'({label_a} vs {label_b})', fontsize=8)
    return fig, ax


def plot_corruption_modes(pattern, q=0.7, figsize=(10.5, 9.5), filepath=None,
                          seed=123,
                          radial_kwargs=None,
                          block_kwargs=None,
                          gradient_kwargs=None):

    # Visualise the four corruption strategies side by side, in four rows:
    # original pattern, flipped-pixel map, corrupted pattern, and local
    # retention-probability map. The same seed is shared by the corruption and the
    # map so that the block patch shown in the last row matches the one flipped.

    from patterns_v2 import corrupt_patterns, corruption_map

    methods = ['random', 'radial', 'block', 'gradient']
    labels = ['Random\n(uniform)', 'Radial\n(centre preserved)',
              'Block\n(patch removed)', 'Gradient\n(top → bottom)']

    kw_radial = {}
    kw_block = {}
    kw_gradient = {'direction': 'vertical'}
    if radial_kwargs:   kw_radial.update(radial_kwargs)
    if block_kwargs:    kw_block.update(block_kwargs)
    if gradient_kwargs: kw_gradient.update(gradient_kwargs)
    kwargs = [{}, kw_radial, kw_block, kw_gradient]

    n = len(methods)
    fig, axes = plt.subplots(4, n, figsize=figsize)
    row_labels = ['Original', 'Flipped pixels', 'Corrupted', 'Retention prob.']

    for col, (m, lbl, kw) in enumerate(zip(methods, labels, kwargs)):
        corrupted = corrupt_patterns([pattern], q, method=m, seed=seed, **kw)[0]
        q_map = corruption_map(pattern.shape, q, method=m, seed=seed, **kw)

        axes[0, col].imshow((pattern + 1) / 2, cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(lbl, fontsize=10)

        plot_pixel_diff(pattern, corrupted, ax=axes[1, col], label_a='orig', label_b='corr')

        axes[2, col].imshow((corrupted + 1) / 2, cmap='gray', vmin=0, vmax=1)

        im = axes[3, col].imshow(q_map, cmap='RdYlGn', vmin=0, vmax=1)

        for r in range(4):
            axes[r, col].set_xticks([]); axes[r, col].set_yticks([])

    for r, rlab in enumerate(row_labels):
        axes[r, 0].set_ylabel(rlab, fontsize=10, fontweight='bold')

    fig.colorbar(im, ax=axes[3, -1], shrink=0.9, label='Retention probability q')
    fig.suptitle(f'Corruption methods  (global q = {q})', fontsize=12, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0.02, 0, 1, 0.98])
    if filepath:
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.show()


def plot_all_results(patterns, corrupted, recovered,
                     model=None, figsize=None, filepath=None,
                     labels=None, show_diff=True):

    # Four-row grid: original | corrupted | recovered | residual error. The last
    # row marks in red the pixels where the recovered pattern still differs from
    # the original. If a model is given, the overlap with the original is shown as
    # a subtitle. Set show_diff=False for the legacy 3-row layout.

    P = len(patterns)
    n_rows = 4 if show_diff else 3
    fw = figsize[0] if figsize else max(8, 2 * P)
    fh = figsize[1] if figsize else (1.6 * n_rows)
    fig, axes = plt.subplots(n_rows, P, figsize=(fw, fh))
    if P == 1:
        axes = axes[:, None]

    row_labels = ['Original', 'Corrupted', 'Recovered']
    if show_diff:
        row_labels.append('Residual\nerror')

    for row, (row_ax, rlab) in enumerate(zip(axes, row_labels)):
        for col, ax in enumerate(row_ax):
            if row == 3:
                plot_pixel_diff(patterns[col], recovered[col], ax=ax,
                                label_a='orig', label_b='rec', show_count=True)
            else:
                if row == 0:
                    img = patterns[col]
                elif row == 1:
                    img = corrupted[col]
                else:
                    img = recovered[col]

                ax.imshow(img.squeeze().reshape(patterns[col].shape), cmap='gray', vmin=-1, vmax=1)
                ax.axis('off')

            if row == 0:
                title = labels[col] if labels else f'Pattern {col}'
                ax.set_title(title, fontsize=9)

            if col == 0:
                ax.annotate(rlab, xy=(0, 0.5), xytext=(-0.15, 0.5),
                            xycoords='axes fraction', textcoords='axes fraction',
                            fontsize=9, ha='right', va='center', rotation=90)

            if row == 2 and model is not None:
                ov = model.overlap(patterns[col], recovered[col])
                ax.set_xlabel(f'ov = {float(ov):.2f}', fontsize=8)

    plt.tight_layout()
    if filepath:
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.show()


def compute_phase_diagram(shape, T_values, alpha_values,
                          n_reps=3, n_therm=3, n_measure=5,
                          seed=0, verbose=True):

    # Sample the equilibrium overlap <m> on a grid in (T, alpha = P/N) using the
    # Monte Carlo update at fixed temperature (no annealing). For each cell the
    # state is initialised on a stored pattern, thermalised for n_therm sweeps,
    # then the overlap with that pattern is time-averaged over n_measure sweeps.

    # Returns a dict with keys 'T_values', 'alpha_values', 'P_values', 'm_grid',
    # 'N'.
    import time
    torch.manual_seed(seed)

    N = shape[0] * shape[1]
    P_values = np.unique(np.maximum(1, np.round(np.asarray(alpha_values) * N).astype(int)))
    alpha_values = P_values / N

    m_grid = np.zeros((len(T_values), len(P_values)))
    t0 = time.time()
    if verbose:
        print(f"Phase diagram grid: {len(T_values)} × {len(P_values)} "
              f"= {len(T_values)*len(P_values)} cells, {n_reps} reps each.")

    for j, P in enumerate(P_values):
        if verbose:
            print(f"  alpha = {P/N:.3f}  (P={P}) ...", end=" ", flush=True)
        for i, T in enumerate(T_values):
            m_runs = []
            for rep in range(n_reps):
                pats = generate_random_patterns(shape, P, seed=seed + rep)
                model = hopfield_model_torch(pats, update_method='montecarlo',
                                             learning_rule='hebb', verbose=False)
                xi1 = torch.as_tensor(pats[0], dtype=torch.float32, device=model.device).flatten()
                state = xi1.clone()

                for _ in range(n_therm):
                    state = model.update(state, steps=1, temperature=T,
                                         alpha=0.0, global_iter=0, schedule='classic')

                m_acc = 0.0
                for _ in range(n_measure):
                    state = model.update(state, steps=1, temperature=T,
                                         alpha=0.0, global_iter=0, schedule='classic')
                    m_acc += float(torch.dot(state, xi1) / N)
                m_runs.append(m_acc / n_measure)
            m_grid[i, j] = np.mean(m_runs)
        if verbose:
            print(f"done  ({time.time()-t0:6.1f}s elapsed)")

    if verbose:
        print(f"\nTotal time: {time.time()-t0:.1f}s")
        print(f"Retrieval region (<m> > 0.5) covers "
              f"{100*np.mean(m_grid > 0.5):.1f}% of the sampled (T,alpha) plane.")

    return {'T_values': T_values, 'alpha_values': alpha_values,
            'P_values': P_values, 'm_grid': m_grid, 'N': N}


def plot_phase_diagram(results, filepath=None, figsize=(8, 5.2),
                       T_c=1.0, alpha_c=0.138, contour_level=0.5):

    # Plot the (T, alpha) phase diagram from a compute_phase_diagram result, with
    # the mean-field reference lines T_c = 1 and alpha_c ~ 0.138 and an empirical
    # <m> = contour_level retrieval boundary.
    T_values = results['T_values']
    alpha_values = results['alpha_values']
    m_grid = results['m_grid']
    N = results['N']

    def _edges(v):
        v = np.asarray(v, dtype=float)
        d = np.diff(v) / 2.0
        return np.concatenate([[v[0] - d[0]], v[:-1] + d, [v[-1] + d[-1]]])

    A_edges, T_edges = _edges(alpha_values), _edges(T_values)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(A_edges, T_edges, m_grid, cmap='RdYlGn', vmin=-1, vmax=1, shading='flat')

    ax.axhline(T_c, color='k', ls='--', lw=1.4,
               label=rf'$T_c = {T_c}$ (mean-field melting)')
    ax.axvline(alpha_c, color='royalblue', ls='--', lw=1.4,
               label=rf'$\alpha_c \approx {alpha_c}$ (mean-field capacity)')

    try:
        ax.contour(alpha_values, T_values, m_grid, levels=[contour_level],
                   colors='black', linewidths=1.6, linestyles='-')
        ax.plot([], [], 'k-', lw=1.6, label=rf'$\langle m\rangle = {contour_level}$ contour')
    except Exception:
        pass

    a_lo, a_hi = alpha_values[0], alpha_values[-1]
    T_lo, T_hi = T_values[0], T_values[-1]
    ax.text(a_lo + 0.02*(a_hi-a_lo), T_hi - 0.15*(T_hi-T_lo),
            'P\n(paramagnetic)', fontsize=10, fontweight='bold', ha='left', va='center')
    ax.text(a_lo + 0.02*(a_hi-a_lo), T_lo + 0.15*(T_hi-T_lo),
            'R\n(retrieval)', fontsize=10, fontweight='bold', ha='left', va='center')
    ax.text(a_hi - 0.15*(a_hi-a_lo), T_lo + 0.15*(T_hi-T_lo),
            'SG\n(spin glass)', fontsize=10, fontweight='bold', ha='left', va='center')

    ax.set_xlabel(r'Storage load  $\alpha = P/N$', fontsize=12)
    ax.set_ylabel(r'Temperature  $T$', fontsize=12)
    ax.set_title(rf'Phase diagram of the Hopfield model  ($N = {N}$)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'Time-averaged overlap  $\langle m\rangle$', fontsize=11)

    plt.tight_layout()
    if filepath:
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.show()


def plot_energy_trace(model, corrupted_patterns, max_iter=100,
                      temperature=0.0, alpha=0.0, schedule='classic',
                      filepath=None, figsize=(7, 3.5)):
    all_energies = []

    for corrupted_pattern in corrupted_patterns:
        pattern = torch.tensor(corrupted_pattern.flatten(), dtype=torch.float32, device=model.device)
        energies = [float(model.energy(pattern.cpu().numpy()))]

        for i in range(max_iter):
            pattern = model.update(pattern.cpu().numpy(), steps=1,
                                   temperature=temperature, alpha=alpha,
                                   global_iter=i, schedule=schedule)
            pattern = torch.tensor(pattern.flatten(), dtype=torch.float32, device=model.device)
            energies.append(float(model.energy(pattern.cpu().numpy())))

        all_energies.append(energies)

    all_energies = np.array(all_energies)
    mean_energy = all_energies.mean(axis=0)

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
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.show()

    return all_energies


def plot_overlap_vs_q(results_dict, n_patterns_list, N, metric='avg_overlap', filepath=None, figsize=None):

    # Plot average overlap vs q for several values of P (number of patterns).
    # results_dict : { P: { q: (avg_overlap, std_overlap) } }

    fig, ax = plt.subplots(figsize=figsize)

    for i, P in enumerate(n_patterns_list):
        color = PALETTE[i % len(PALETTE)]
        data = results_dict[P]
        qs = sorted(data.keys())
        # data[q] may be a (mean, std) tuple or a result dict
        avgs = np.array([data[q][0] if isinstance(data[q], tuple) else data[q][metric] for q in qs])
        stds = np.array([data[q][1] if isinstance(data[q], tuple) else data[q].get('std_overlap', 0) for q in qs])

        ax.plot(qs, avgs, marker='o', markersize=5, linewidth=2, color=color, label=f'P = {P}')
        ax.fill_between(qs, np.clip(avgs - stds, -1, 1), np.clip(avgs + stds, -1, 1),
                        color=color, alpha=0.15)

    ax.axhline(0, color='grey', linestyle=':', linewidth=1)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1.2, label='q = 0.5 (50 % noise)')
    ax.set_xlabel('Retention probability q', fontsize=12)
    ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
    ax.set_title(f'Overlap vs corruption level  (N = {N})', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1); ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    if filepath:
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.show()


def plot_overlap_vs_P(results_dict, q_list, N, filepath=None, figsize=(8, 4.5)):

    # Plot average overlap vs P for several values of q, with theoretical limit.
    fig, ax = plt.subplots(figsize=figsize)
    theoretical_limit = 0.138 * N

    for i, q in enumerate(q_list):
        color = PALETTE[i % len(PALETTE)]
        data = results_dict[q]
        Ps = sorted(data.keys())
        avgs = np.array([data[P][0] for P in Ps])
        stds = np.array([data[P][1] for P in Ps])

        ax.plot(Ps, avgs, marker='o', markersize=5, linewidth=2, color=color, label=f'q = {q:.2f}')
        ax.fill_between(Ps, np.clip(avgs - stds, -1, 1), np.clip(avgs + stds, -1, 1),
                        color=color, alpha=0.15)

    ax.axvline(theoretical_limit, color='crimson', linestyle='--', linewidth=2,
               label=f'Theoretical limit  $P_c ≈ 0.138N$ = {theoretical_limit:.0f}')
    ax.set_xlabel('Number of stored patterns  P', fontsize=12)
    ax.set_ylabel('Average overlap  $m$', fontsize=12)
    ax.set_title(f'Storage capacity  (N = {N})', fontsize=13, fontweight='bold')
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    if filepath:
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.show()


def plot_capacity_heatmap(results_dict, q_list, P_list, N, filepath=None, figsize=(9, 5)):

    # Heatmap of avg_overlap on axes (q, P).
    import seaborn as sns

    data = np.array([[results_dict[q][P][0] for P in P_list] for q in q_list])
    fig, ax = plt.subplots(figsize=figsize)

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    sns.heatmap(data, xticklabels=P_list,
                yticklabels=[f'{q:.2f}' for q in q_list],
                cmap='RdYlGn', norm=norm,
                linewidths=0.3, linecolor='white',
                cbar_kws={'label': 'Average overlap $m$'}, ax=ax)

    for j, P in enumerate(P_list):
        if P > 0.138 * N:
            ax.axvline(j, color='royalblue', linewidth=1.5, linestyle='--')
            ax.text(j + 0.1, 0.3, f'$P_c≈{0.138*N:.0f}$', color='royalblue', fontsize=8, va='top')
            break

    ax.set_xlabel('Number of stored patterns  P', fontsize=12)
    ax.set_ylabel('Retention probability  q', fontsize=12)
    ax.set_title(f'Average overlap heatmap  (N = {N})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filepath:
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.show()
