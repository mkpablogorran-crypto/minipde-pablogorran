import numpy as np
import pytest

from minipde.mesh import uniform_grid
from minipde.operators import first_derivative, second_derivative


def test_derivadas_de_una_constante_son_cero():
    x = uniform_grid(0.0, 1.0, 11)
    u = np.full_like(x, 3.0)
    assert np.allclose(first_derivative(x, u), 0.0)
    assert np.allclose(second_derivative(x, u), 0.0)


def test_primera_derivada_de_recta_es_exacta():
    x = uniform_grid(-1.0, 2.0, 7)
    assert np.allclose(first_derivative(x, 2 * x - 1), 2.0)


def test_segunda_derivada_de_x2_es_exacta():
    x = uniform_grid(-1.0, 2.0, 7)
    assert np.allclose(second_derivative(x, x**2), 2.0)


def test_no_modifica_las_entradas():
    x = uniform_grid(0.0, 1.0, 11)
    u = np.sin(x)
    x0, u0 = x.copy(), u.copy()
    first_derivative(x, u)
    second_derivative(x, u)
    assert np.array_equal(x, x0)
    assert np.array_equal(u, u0)


def test_first_derivative_is_second_order():
    errors = []
    for n in (41, 81, 161):
        x = uniform_grid(0.0, 2 * np.pi, n)
        du = first_derivative(x, np.sin(x))
        errors.append(np.max(np.abs(du - np.cos(x))))
    p = np.log2(np.array(errors[:-1]) / errors[1:])
    assert np.all(p > 1.8)


def test_second_derivative_is_second_order():
    errors = []
    for n in (41, 81, 161):
        x = uniform_grid(0.0, 2 * np.pi, n)
        d2u = second_derivative(x, np.sin(x))
        errors.append(np.max(np.abs(d2u + np.sin(x))))
    p = np.log2(np.array(errors[:-1]) / errors[1:])
    assert np.all(p > 1.8)


@pytest.mark.parametrize(
    "x, u",
    [
        (np.array([0.0, 1.0]), np.array([0.0, 1.0])),            # pocos nodos
        (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0])),       # tamaños distintos
        (np.array([0.0, 2.0, 1.0]), np.array([0.0, 1.0, 2.0])),  # no creciente
        (np.array([0.0, 1.0, 3.0]), np.array([0.0, 1.0, 2.0])),  # no uniforme
        (np.array([0.0, 1.0, 2.0]), np.array([0.0, np.nan, 2.0])),  # no finito
    ],
)
def test_entradas_invalidas(x, u):
    with pytest.raises(ValueError):
        first_derivative(x, u)


def test_segunda_derivada_necesita_4_nodos():
    x = uniform_grid(0.0, 1.0, 3)
    with pytest.raises(ValueError):
        second_derivative(x, x**2)