"""
patterns.py
===========
Utility functions for the Hopfield network project.

Contents
--------
generate_random_patterns   – generate P random {+1,−1} arrays of given shape
corrupt_patterns           – corrupt a list of patterns with different strategies
get_mnist_patterns         – compute MNIST class prototypes (naive and upscaled)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms


# ──────────────────────────────────────────────────────────────────────────────
# 1.  RANDOM PATTERN GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def generate_random_patterns(shape, n_patterns, seed=None):
    """
    Generate n_patterns random binary patterns of given shape.

    Parameters
    ----------
    shape      : tuple, e.g. (10, 10) or (784,)
    n_patterns : int
    seed       : int or None  (for reproducibility)

    Returns
    -------
    list of np.ndarray, each shape=`shape`, values in {+1, −1}
    """
    rng = np.random.default_rng(seed)
    return [rng.choice([-1, 1], size=shape).astype(np.float32)
            for _ in range(n_patterns)]


# ──────────────────────────────────────────────────────────────────────────────
# 2.  PATTERN CORRUPTION
# ──────────────────────────────────────────────────────────────────────────────

def _make_flip_mask_random(shape, q, rng):
    """Uniform random flips: each pixel flipped with probability (1−q)."""
    return rng.random(shape) > q          # True → flip


def _make_flip_mask_radial(shape, q_center, q_edge, rng):
    """
    Radial corruption: pixels near the centre are well preserved (q≈q_center),
    pixels near the edge are heavily corrupted (q≈q_edge).

    The local retention probability interpolates linearly with normalised radius.
    """
    if len(shape) != 2:
        raise ValueError("'radial' corruption requires 2-D patterns.")
    H, W = shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    max_r  = np.sqrt(cy**2 + cx**2)

    ys = np.arange(H)[:, None]
    xs = np.arange(W)[None, :]
    r  = np.sqrt((ys - cy)**2 + (xs - cx)**2) / max_r   # in [0, 1]

    # local retention probability: high at centre, low at edge
    q_local = q_center + (q_edge - q_center) * r        # shape (H, W)
    return rng.random(shape) > q_local


def _make_flip_mask_block(shape, q, block_fraction, rng):
    """
    Block corruption: a contiguous rectangular region is fully corrupted;
    the rest of the image is left intact.

    `block_fraction` controls the side length of the block as a fraction of
    the image dimensions (default 0.5 → 50 % of each side).
    """
    mask = np.zeros(shape, dtype=bool)
    if len(shape) == 2:
        H, W = shape
        bH = max(1, int(H * block_fraction))
        bW = max(1, int(W * block_fraction))
        r0 = rng.integers(0, max(1, H - bH + 1))
        c0 = rng.integers(0, max(1, W - bW + 1))
        mask[r0:r0 + bH, c0:c0 + bW] = True
    else:
        n = shape[0]
        b = max(1, int(n * block_fraction))
        i0 = rng.integers(0, max(1, n - b + 1))
        mask[i0:i0 + b] = True
    return mask


def _make_flip_mask_gradient(shape, q_min, q_max, direction, rng):
    """
    Gradient corruption: retention probability varies linearly from q_max
    to q_min along a chosen axis.

    direction : 'horizontal' (left → right) or 'vertical' (top → bottom)
    """
    if len(shape) != 2:
        raise ValueError("'gradient' corruption requires 2-D patterns.")
    H, W = shape
    if direction == 'horizontal':
        t = np.linspace(0, 1, W)[None, :]           # (1, W)
        t = np.broadcast_to(t, (H, W))
    elif direction == 'vertical':
        t = np.linspace(0, 1, H)[:, None]           # (H, 1)
        t = np.broadcast_to(t, (H, W))
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'.")

    q_local = q_max + (q_min - q_max) * t           # (H, W)
    return rng.random(shape) > q_local


# ─── public API ───────────────────────────────────────────────────────────────

def corrupt_patterns(patterns, q, method='random', seed=None, **kwargs):
    """
    Corrupt a list of patterns.

    Parameters
    ----------
    patterns : list of np.ndarray, values in {+1, −1}
    q        : float in [0, 1]
                 - for 'random': global retention probability
                 - for 'radial': q_center (q_edge defaults to 1−q_center via kwargs)
                 - for 'gradient': q_max (the well-preserved end)
                 - ignored for 'block' (block pixels always flipped)
    method   : str
                 'random'   – spatially uniform noise
                 'radial'   – grows from centre outward
                 'block'    – one contiguous rectangular patch
                 'gradient' – left-to-right or top-to-bottom gradient
    seed     : int or None
    **kwargs :
        radial   → q_edge (float, default 1−q)
        block    → block_fraction (float in (0,1], default 0.5)
        gradient → direction ('horizontal'|'vertical', default 'vertical'),
                   q_min (float, default 1−q)

    Returns
    -------
    list of np.ndarray, same shapes as input, values in {+1, −1}
    """
    rng = np.random.default_rng(seed)
    corrupted = []

    for p in patterns:
        shape = p.shape

        if method == 'random':
            mask = _make_flip_mask_random(shape, q, rng)

        elif method == 'radial':
            q_edge = kwargs.get('q_edge', 1.0 - q)
            mask   = _make_flip_mask_radial(shape, q_center=q, q_edge=q_edge, rng=rng)

        elif method == 'block':
            frac = kwargs.get('block_fraction', 0.5)
            mask = _make_flip_mask_block(shape, q, frac, rng)

        elif method == 'gradient':
            direction = kwargs.get('direction', 'vertical')
            q_min     = kwargs.get('q_min', 1.0 - q)
            mask      = _make_flip_mask_gradient(shape, q_min=q_min, q_max=q,
                                                  direction=direction, rng=rng)
        else:
            raise ValueError(f"Unknown corruption method: '{method}'. "
                             f"Choose from: random, radial, block, gradient.")

        c = p.copy()
        c[mask] *= -1
        corrupted.append(c)

    return corrupted


def corruption_map(shape, q, method='random', seed=None, **kwargs):
    """
    Return the local effective retention-probability map for a given method.
    Useful for visualising what each corruption mode looks like before applying it.

    Returns
    -------
    np.ndarray of shape `shape`, values in [0,1]  (1 = always kept, 0 = always flipped)
    """
    rng = np.random.default_rng(seed)
    if method == 'random':
        return np.full(shape, q, dtype=np.float32)
    elif method == 'radial':
        q_edge = kwargs.get('q_edge', 1.0 - q)
        H, W   = shape
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        max_r  = np.sqrt(cy**2 + cx**2)
        ys = np.arange(H)[:, None]; xs = np.arange(W)[None, :]
        r  = np.sqrt((ys - cy)**2 + (xs - cx)**2) / max_r
        # return (q + (q_edge - q) * r).astype(np.float32)
        return (1.0 + (q_edge - 1.0) * r).astype(np.float32)
    elif method == 'gradient':
        direction = kwargs.get('direction', 'vertical')
        q_min     = kwargs.get('q_min', 1.0 - q)
        H, W = shape
        t = (np.linspace(0, 1, H)[:, None] if direction == 'vertical'
             else np.linspace(0, 1, W)[None, :])
        return (q + (q_min - q) * np.broadcast_to(t, shape)).astype(np.float32)
    elif method == 'block':
        return np.full(shape, q, dtype=np.float32)   # approximate; block is random position
    else:
        raise ValueError(f"Unknown method '{method}'.")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  MNIST PROTOTYPE GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def _binarize_fixed(img, threshold=0.5):
    """Fixed global threshold: pixel > threshold → +1, else −1."""
    return np.where(img > threshold, 1.0, -1.0).astype(np.float32)


def get_mnist_patterns(
    N=28,
    mode='majority_vote',
    upscale_factor=1,
    digits=None,
    data_root='./mnist_data',
    verbose=True,
):
    """
    Compute one binary prototype per MNIST digit class.

    The function exposes two strategies to understand the effect of resolution:

    mode='majority_vote'  (default)
        For each pixel position and each training image, binarise the pixel
        independently (+1 if > 0.5) and take the *majority vote* across all
        images of that class. This is fast but can create holes (isolated –1
        pixels in a mostly +1 region) when the training images are noisy.

    mode='mean_then_binarize'
        Average the raw grayscale images first, then binarise the mean with a
        fixed threshold.  Produces smoother prototypes but loses fine detail.

    upscale_factor : int ≥ 1
        Before binarising, upscale each N×N image by this factor using bilinear
        interpolation, compute the prototype at (N·upscale_factor)² pixels, then
        downscale back to N×N.  The averaging in high-res space fills in holes
        that appear at low resolution.  Useful to demonstrate that the naive
        majority-vote prototype is fragile at small N.

    Parameters
    ----------
    N             : int   – output prototype side length in pixels (default 28)
    mode          : str   – 'majority_vote' or 'mean_then_binarize'
    upscale_factor: int   – upscaling factor applied before binarisation (≥1)
    digits        : list  – which digit classes to include (default: 0-9)
    data_root     : str   – path for MNIST download cache
    verbose       : bool

    Returns
    -------
    list of np.ndarray, length = len(digits), each shape (N, N), values in {+1, −1}
    """
    if digits is None:
        digits = list(range(10))

    if verbose:
        print(f"Loading MNIST (N={N}, mode={mode}, upscale={upscale_factor}×) …")

    transform = transforms.Compose([transforms.ToTensor()])
    dataset   = datasets.MNIST(root=data_root, train=True,
                               download=True, transform=transform)

    Hi = N * upscale_factor   # high-res side length

    # Per-class accumulators at high resolution
    sums   = {d: np.zeros((Hi, Hi), dtype=np.float64) for d in digits}
    counts = {d: 0 for d in digits}

    resize_hi = transforms.Resize((Hi, Hi), antialias=True)

    for img_t, label in dataset:
        if label not in digits:
            continue
        img_hi = resize_hi(img_t).squeeze(0).numpy()   # (Hi, Hi) in [0,1]

        if mode == 'majority_vote':
            # binarise at high-res, then accumulate the vote
            sums[label]   += _binarize_fixed(img_hi)
        elif mode == 'mean_then_binarize':
            # accumulate raw grayscale at high-res
            sums[label]   += img_hi
        else:
            raise ValueError(f"Unknown mode '{mode}'.")

        counts[label] += 1

    prototypes = []
    for d in digits:
        if counts[d] == 0:
            prototypes.append(np.ones((N, N), dtype=np.float32))
            continue

        mean_hi = sums[d] / counts[d]       # (Hi, Hi)

        if mode == 'majority_vote':
            # majority vote: sign of average binary votes
            proto_hi = np.sign(mean_hi).astype(np.float32)
            proto_hi[proto_hi == 0] = 1.0   # break ties
        else:
            # mean_then_binarize: threshold the averaged grayscale
            proto_hi = _binarize_fixed(mean_hi).astype(np.float32)

        if upscale_factor > 1:
            # downscale back to N×N via average pooling (fills holes)
            t = torch.from_numpy(proto_hi).unsqueeze(0).unsqueeze(0)
            t_down = F.avg_pool2d(t, kernel_size=upscale_factor,
                                   stride=upscale_factor)
            mean_lo = t_down.squeeze().numpy()   # (N, N) in (-1, 1)
            proto   = np.sign(mean_lo).astype(np.float32)
            proto[proto == 0] = 1.0
        else:
            proto = proto_hi

        prototypes.append(proto)

    if verbose:
        print(f"  Done. Counts per class: { {d: counts[d] for d in digits} }")
    return prototypes
