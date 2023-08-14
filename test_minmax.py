import numpy as np
import jax.numpy as jnp
import pytest

from minmax import minmax_kernel, minmax_kernel_l1, minmax_random_features


@pytest.mark.parametrize("kernel_fn", [minmax_kernel, minmax_kernel_l1])
def test_scalar_kernel(kernel_fn):
    result = kernel_fn(jnp.array([1.0, 2.0, 3.0]), jnp.array([3.0, 2.0, 1.0]))
    assert jnp.isclose(result, 0.5).all()


def test_random_features():
    # Compute many random features

    x1 = jnp.array([1.0, 2.0, 3.0])
    x2 = jnp.array([3.0, 2.0, 1.0])
    modulo_value = 8

    x1_features = []
    x2_features = []

    for _ in range(1_000):
        r = jnp.array(-np.log(np.random.rand(3)) - np.log(np.random.rand(3)))
        c = jnp.array(-np.log(np.random.rand(3)) - np.log(np.random.rand(3)))
        xi = jnp.array(np.random.randint(0, 2, size=(3, modulo_value)) * 2 - 1)
        beta = np.random.rand(3)

        x1_features.append(
            minmax_random_features(
                x1, xi=xi, r=r, c=c, beta=beta, modulo_value=modulo_value
            )
        )

        x2_features.append(
            minmax_random_features(
                x2, xi=xi, r=r, c=c, beta=beta, modulo_value=modulo_value
            )
        )

    x1_features = jnp.array(x1_features)
    x2_features = jnp.array(x2_features)

    # Compute inner product between features
    rf_prod = jnp.mean(
        x1_features * x2_features
    )  # mean is because the features are not properly normalized

    # Test: is random feature inner product correct?
    assert jnp.all(jnp.abs(rf_prod - 0.5) < 0.05)
