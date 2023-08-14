"""Code related to minmax kernel and its random features."""

import jax.numpy as jnp


def minmax_kernel(x, y):
    r"""
    Computes the following kernel between two non-negative vectors:

    \frac{\sum_i \min(x_i, y_i)}{\sum_i \max(x_i, y_i)}

    This is just designed for scalars.
    """

    return jnp.sum(jnp.minimum(x, y)) / jnp.sum(jnp.maximum(x, y))


def minmax_kernel_l1(x, y):
    r"""
    Alternative implementation above using the L1 distance formula
    given in Ioffe (2010):

    \frac{|x|+|y|-|x-y|}{|x|+|y|+|x-y|}

    Where all norms are L1 norms.
    """

    return (jnp.sum(x) + jnp.sum(y) - jnp.sum(jnp.abs(x - y))) / (
        jnp.sum(x) + jnp.sum(y) + jnp.sum(jnp.abs(x - y))
    )


def minmax_random_features(x, xi, r, c, beta, modulo_value=8):
    r"""
    Computes hash-based random features for minmax kernel.
    This implementation is not vectorized at all (not for inputs, and not for random features).

    See equation 6 from http://arxiv.org/abs/2306.14809 (first version).
    The specific hash is the Ioffe (2010) hash, given in equations 26-30 of arXiv paper above.
    Because this hash returns 2 integers, xi is a 2D array of shape (D, modulo value) rather than a 1D vector
    as in the paper.
    """

    # Compute the hash value
    # CWS variables
    t = jnp.floor(jnp.log(x) / r + beta)  # shape D (same as input x)
    ln_y = r * (t - beta)  # also shape D
    ln_a = jnp.log(c) - ln_y - r  # also shape D

    # argmin
    a_argmin = jnp.argmin(ln_a)  # this only works for 1D inputs, vectorizing will break
    t_selected = int(t[a_argmin])

    # Use this to index xi
    return xi[a_argmin, t_selected % modulo_value]
