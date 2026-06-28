"""
Pattern generation and corruption helpers for the Hopfield project.

generate_random_patterns - generate P random {+1,-1} arrays of given shape
corrupt_patterns          - corrupt a list of patterns with different strategies
corruption_map            - local retention-probability map for a given method
get_mnist_patterns        - compute MNIST class prototypes
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms


def generate_random_patterns(shape, n_patterns, seed=None):
    rng = np.random.default_rng(seed)
    return [rng.choice([-1, 1], size=shape).astype(np.float32) for _ in range(n_patterns)]


def _make_flip_mask_random(shape, q, rng):
# Uniform random flips: each pixel flipped with probability (1-q).
    return rng.random(shape) > q


def _make_flip_mask_radial(shape, q_center, q_edge, rng, core_fraction=0.4):

    # Radial corruption: a central disk of radius `core_fraction` (relative to the
    # maximum radius) is preserved with probability q_center; outside the core the
    # retention probability decays linearly to q_edge at the corners.

    # core_fraction = 0 gives a linear ramp from the centre, 0.4 keeps the inner
    # 40% fully preserved, 1 keeps the whole image at q_center.
    if len(shape) != 2:
        raise ValueError("'radial' corruption requires 2-D patterns.")
    H, W = shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    max_r = np.sqrt(cy**2 + cx**2)

    ys = np.arange(H)[:, None]
    xs = np.arange(W)[None, :]
    r = np.sqrt((ys - cy)**2 + (xs - cx)**2) / max_r

    if core_fraction >= 1.0:
        q_local = np.full(shape, q_center, dtype=np.float64)
    else:
        t = np.clip((r - core_fraction) / (1.0 - core_fraction), 0.0, 1.0)
        q_local = q_center + (q_edge - q_center) * t
    return rng.random(shape) > q_local


def _radial_retention_for_q(shape, q):

    # Circular retention map tied to q: the centre is fully preserved and fades to
    # 0 towards the edges, with the radius tuned so that the mean retention equals
    # q (flipped fraction = 1-q). Works for any q in [0,1], unlike the
    # q_center/q_edge parametrisation.

    # retention(r) = clip(1 - r/theta, 0, 1), with theta found by bisection (the
    # mean is monotonic in theta).
    if len(shape) != 2:
        raise ValueError("'radial' corruption requires 2-D patterns.")
    H, W = shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    max_r = np.sqrt(cy**2 + cx**2)
    ys = np.arange(H)[:, None]; xs = np.arange(W)[None, :]
    r = np.sqrt((ys - cy)**2 + (xs - cx)**2) / max_r
    q = float(np.clip(q, 0.0, 1.0))
    lo, hi = 1e-6, 1e6
    for _ in range(60):
        theta = 0.5 * (lo + hi)
        if np.clip(1.0 - r / theta, 0.0, 1.0).mean() < q:
            lo = theta
        else:
            hi = theta
    return np.clip(1.0 - r / theta, 0.0, 1.0).astype(np.float32)


def _make_flip_mask_block(shape, q, block_fraction, rng):

    # Block corruption: a contiguous rectangular region is fully corrupted and the
    # rest of the image is left intact. `block_fraction` is the side length of the
    # block as a fraction of the image dimensions.

    mask = np.zeros(shape, dtype=bool)
    if len(shape) == 2:
        H, W = shape
        bH = max(1, int(round(H * block_fraction)))
        bW = max(1, int(round(W * block_fraction)))
        r0 = rng.integers(0, max(1, H - bH + 1))
        c0 = rng.integers(0, max(1, W - bW + 1))
        mask[r0:r0 + bH, c0:c0 + bW] = True
    else:
        n = shape[0]
        b = max(1, int(round(n * block_fraction)))
        i0 = rng.integers(0, max(1, n - b + 1))
        mask[i0:i0 + b] = True
    return mask


def _make_flip_mask_gradient(shape, q_min, q_max, direction, rng):

    # Gradient corruption: retention probability varies linearly from q_max to
    # q_min along the chosen axis ('horizontal' or 'vertical').

    if len(shape) != 2:
        raise ValueError("'gradient' corruption requires 2-D patterns.")
    H, W = shape
    if direction == 'horizontal':
        t = np.linspace(0, 1, W)[None, :]
        t = np.broadcast_to(t, (H, W))
    elif direction == 'vertical':
        t = np.linspace(0, 1, H)[:, None]
        t = np.broadcast_to(t, (H, W))
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'.")

    q_local = q_max + (q_min - q_max) * t
    return rng.random(shape) > q_local


def corrupt_patterns(patterns, q, method='random', seed=None, **kwargs):

    # Corrupt a list of patterns (values in {+1, -1}).

    # q is the global retention probability for 'random', q_center for 'radial',
    # q_max for 'gradient', and is ignored for 'block'. Extra keyword arguments:
    # radial -> q_edge; block -> block_fraction; gradient -> direction, q_min.

    rng = np.random.default_rng(seed)
    corrupted = []

    for p in patterns:
        shape = p.shape

        if method == 'random':
            mask = _make_flip_mask_random(shape, q, rng)

        elif method == 'radial':
            if 'q_center' in kwargs or 'q_edge' in kwargs:
                # explicit (old) behaviour: core_fraction plateau + ramp
                q_center = kwargs.get('q_center', q)
                q_edge = kwargs.get('q_edge', 1.0 - q_center)
                core_fraction = kwargs.get('core_fraction', 0.4)
                mask = _make_flip_mask_radial(shape, q_center=q_center, q_edge=q_edge, rng=rng, core_fraction=core_fraction)
            else:
                # default tied to q: flipped fraction = 1-q, centre preserved
                ret = _radial_retention_for_q(shape, q)
                mask = rng.random(shape) > ret

        elif method == 'block':
            # Block whose area is the flipped fraction (1-q): small q -> large
            # block, large q -> small block. In 2D the area is the side squared,
            # so side = sqrt(1-q); in 1D side = 1-q.
            default_side = np.sqrt(max(0.0, 1.0 - q)) if len(shape) == 2 else max(0.0, 1.0 - q)
            frac = kwargs.get('block_fraction', default_side)
            mask = _make_flip_mask_block(shape, q, frac, rng)

        elif method == 'gradient':
            # Ramp whose mean retention is q (flipped fraction = 1-q), reaching
            # full retention (1) at the preserved edge when q >= 0.5 and full
            # corruption (0) at the opposite edge when q <= 0.5.
            direction = kwargs.get('direction', 'vertical')
            w = min(q, 1.0 - q)
            q_max = kwargs.get('q_max', q + w)
            q_min = kwargs.get('q_min', q - w)
            mask = _make_flip_mask_gradient(shape, q_min=q_min, q_max=q_max,
                                            direction=direction, rng=rng)
        else:
            raise ValueError(f"Unknown corruption method: '{method}'. "
                             f"Choose from: random, radial, block, gradient.")

        c = p.copy()
        c[mask] *= -1
        corrupted.append(c)

    return corrupted


def corruption_map(shape, q, method='random', seed=None, **kwargs):

    # Return the local retention-probability map for a given method (values in
    # [0,1], 1 = always kept, 0 = always flipped). Useful for visualising what
    # each corruption mode looks like before applying it.
    rng = np.random.default_rng(seed)
    if method == 'random':
        return np.full(shape, q, dtype=np.float32)
    elif method == 'radial':
        if not ('q_center' in kwargs or 'q_edge' in kwargs):
            return _radial_retention_for_q(shape, q)
        q_center = kwargs.get('q_center', q)
        q_edge = kwargs.get('q_edge', 1.0 - q_center)
        core_fraction = kwargs.get('core_fraction', 0.4)
        H, W = shape
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        max_r = np.sqrt(cy**2 + cx**2)
        ys = np.arange(H)[:, None]; xs = np.arange(W)[None, :]
        r = np.sqrt((ys - cy)**2 + (xs - cx)**2) / max_r
        if core_fraction >= 1.0:
            return np.full(shape, q_center, dtype=np.float32)
        t = np.clip((r - core_fraction) / (1.0 - core_fraction), 0.0, 1.0)
        return (q_center + (q_edge - q_center) * t).astype(np.float32)
    elif method == 'gradient':
        direction = kwargs.get('direction', 'vertical')
        w = min(q, 1.0 - q)
        q_hi = kwargs.get('q_max', q + w)
        q_lo = kwargs.get('q_min', q - w)
        H, W = shape
        t = (np.linspace(0, 1, H)[:, None] if direction == 'vertical'
             else np.linspace(0, 1, W)[None, :])
        return (q_hi + (q_lo - q_hi) * np.broadcast_to(t, shape)).astype(np.float32)
    elif method == 'block':
        # Retention 0 inside the block (random), 1 outside. The block area is
        # 1-q (side = sqrt(1-q) in 2D, 1-q in 1D). Using the same rng as the
        # corruption means that, with the same seed, the patch shown here matches
        # the one actually flipped.
        retention = np.ones(shape, dtype=np.float32)
        if len(shape) == 2:
            H, W = shape
            side = kwargs.get('block_fraction', np.sqrt(max(0.0, 1.0 - q)))
            bH = max(1, int(round(H * side)))
            bW = max(1, int(round(W * side)))
            r0 = rng.integers(0, max(1, H - bH + 1))
            c0 = rng.integers(0, max(1, W - bW + 1))
            retention[r0:r0 + bH, c0:c0 + bW] = 0.0
        else:
            n = shape[0]
            side = kwargs.get('block_fraction', max(0.0, 1.0 - q))
            b = max(1, int(round(n * side)))
            i0 = rng.integers(0, max(1, n - b + 1))
            retention[i0:i0 + b] = 0.0
        return retention
    else:
        raise ValueError(f"Unknown method '{method}'.")


def _binarize_fixed(img, threshold=0.5):
    """Fixed global threshold: pixel > threshold -> +1, else -1."""
    return np.where(img > threshold, 1.0, -1.0).astype(np.float32)


def get_mnist_patterns(
    N=28,
    mode='majority_vote',
    upscale_factor=1,
    digits=None,
    data_root='./mnist_data',
    verbose=True,
):

    # Compute one binary prototype per MNIST digit class.

    # Two strategies are available to study the effect of resolution:

    # mode='majority_vote' (default)
    #     Binarise each training image pixel independently (+1 if > 0.5) and take
    #     the majority vote across all images of the class. Fast, but can create
    #     holes when the training images are noisy.

    # mode='mean_then_binarize'
    #     Average the raw grayscale images first, then binarise the mean with a
    #     fixed threshold. Smoother prototypes, but loses fine detail.

    # upscale_factor upscales each N*N image before binarising, computes the
    # prototype at high resolution, then downscales back to N*N; the averaging in
    # high-res space fills in the holes that appear at low resolution.

    # Returns a list of (N, N) arrays with values in {+1, -1}, one per digit.

    if digits is None:
        digits = list(range(10))

    if verbose:
        print(f"Loading MNIST (N={N}, mode={mode}, upscale={upscale_factor}×) …")

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root=data_root, train=True,
                             download=True, transform=transform)

    Hi = N * upscale_factor

    sums = {d: np.zeros((Hi, Hi), dtype=np.float64) for d in digits}
    counts = {d: 0 for d in digits}

    resize_hi = transforms.Resize((Hi, Hi), antialias=True)

    for img_t, label in dataset:
        if label not in digits:
            continue
        img_hi = resize_hi(img_t).squeeze(0).numpy()

        if mode == 'majority_vote':
            sums[label] += _binarize_fixed(img_hi)
        elif mode == 'mean_then_binarize':
            sums[label] += img_hi
        else:
            raise ValueError(f"Unknown mode '{mode}'.")

        counts[label] += 1

    prototypes = []
    for d in digits:
        if counts[d] == 0:
            prototypes.append(np.ones((N, N), dtype=np.float32))
            continue

        mean_hi = sums[d] / counts[d]

        if mode == 'majority_vote':
            proto_hi = np.sign(mean_hi).astype(np.float32)
            proto_hi[proto_hi == 0] = 1.0   # break ties
        else:
            proto_hi = _binarize_fixed(mean_hi).astype(np.float32)

        if upscale_factor > 1:
            # downscale back to N*N via average pooling (fills holes)
            t = torch.from_numpy(proto_hi).unsqueeze(0).unsqueeze(0)
            t_down = F.avg_pool2d(t, kernel_size=upscale_factor, stride=upscale_factor)
            mean_lo = t_down.squeeze().numpy()
            proto = np.sign(mean_lo).astype(np.float32)
            proto[proto == 0] = 1.0
        else:
            proto = proto_hi

        prototypes.append(proto)

    if verbose:
        print(f"  Done. Counts per class: { {d: counts[d] for d in digits} }")
    return prototypes
