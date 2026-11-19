import numpy as np
import pytest

from minipde.mesh import uniform_grid


def test_uniform_grid_extremos_y_tamano():
    x = uniform_grid(0.0, 2.0, 5)
    assert x.shape == (5,)
    assert x[0] == 0.0
    assert x[-1] == 2.0


def test_uniform_grid_equiespaciada():
    x = uniform_grid(-1.0, 3.0, 9)
    h = np.diff(x)
    assert np.all(h > 0)
    assert np.allclose(h, h[0])
    assert np.isclose(h[0], 0.5)


@pytest.mark.parametrize(
    "a, b, n",
    [
        (0.0, 1.0, 1),          # muy pocos puntos
        (1.0, 1.0, 5),          # intervalo vacío
        (2.0, 1.0, 5),          # extremos al revés
        (0.0, np.inf, 5),       # extremo no finito
    ],
)
def test_uniform_grid_entradas_invalidas(a, b, n):
    with pytest.raises(ValueError):
        uniform_grid(a, b, n)


def test_uniform_grid_n_no_entero_da_typeerror():
    with pytest.raises(TypeError):
        uniform_grid(0.0, 1.0, 3.5)