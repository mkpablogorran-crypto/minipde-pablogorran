# MiniPDE

Proyecto transversal de la asignatura **Métodos Numéricos en Ecuaciones
en Derivadas Parciales**.

## Objetivo

Desarrollar progresivamente un pequeño solver modular para ecuaciones
en derivadas parciales, aplicando:

- discretización espacial;
- condiciones de contorno;
- resolución de sistemas lineales y no lineales;
- integración temporal;
- análisis de estabilidad;
- verificación numérica;
- pruebas automatizadas;
- control de versiones con Git.

## Evolución prevista

| Versión | Contenido |
|---|---|
| v0.1 | Entorno, estructura, punto flotante y pruebas |
| v0.2 | Operadores de diferencias finitas |
| v0.3 | Ecuación de Poisson y condiciones de contorno |
| v0.4 | Solvers lineales y no lineales |
| v0.5 | Integradores temporales |
| v0.6 | Ecuación del calor |
| v0.7 | Ecuación de advección |
| v1.0 | Verificación, documentación y proyecto final |

## Instalación

### Crear el entorno virtual

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## v0.2 · Operadores de diferencias finitas

- `minipde.mesh.uniform_grid(a, b, n)`: malla 1D equiespaciada con n nodos en [a, b].
- `minipde.operators.first_derivative(x, u)`: u' de orden 2 (centrada en el interior, adelantada/atrasada en los extremos).
- `minipde.operators.second_derivative(x, u)`: u'' de orden 2 (mismo esquema).
- `minipde.verification.observed_order(h, error)`: orden de convergencia observado entre mallas.

Estudio de convergencia (u = sin x en [0, 2π]):

```powershell
python examples/derivative_convergence.py
```

## v0.3 · Poisson 1D con condiciones de Dirichlet

Resuelve

```text
-u''(x) = f(x)  en [a, b],   u(a) = ua,   u(b) = ub
```

con diferencias finitas centradas de segundo orden en malla uniforme y
`np.linalg.solve`.

### Módulo `src/minipde/poisson.py`

- `assemble_poisson_1d(x, f, ua=0.0, ub=0.0)`: monta el sistema `A u = b`.
  - Filas interiores: `(-u[i-1] + 2 u[i] - u[i+1]) / h**2 = f(x[i])`.
  - Filas 0 y N: se sustituye la ecuación por la condición de Dirichlet
    (`u[0] = ua`, `u[N] = ub`).
  - Lanza `ValueError` si hay menos de 3 puntos o la malla no es equiespaciada.
- `solve_poisson_1d(x, f, ua=0.0, ub=0.0)`: resuelve el sistema y devuelve la
  solución discreta `u` y el residuo `||A u - b||_inf`.

Tests en `tests/test_poisson.py`: forma de la matriz y filas de frontera,
stencil `[-1, 2, -1] / h^2`, exactitud para una solución cuadrática, residuo
pequeño, orden observado ≈ 2 y errores de entrada.

### Ejemplo A · solución manufacturada `u(x) = sin(pi x)`

`f(x) = pi^2 sin(pi x)`, `u(0) = u(1) = 0`.

```powershell
python examples/poisson_dirichlet.py
```

|   N |       h | error L∞   | residuo    | orden |
|----:|--------:|-----------:|-----------:|------:|
|  11 | 0.10000 | 8.2654e-03 | 5.3291e-15 |   –   |
|  21 | 0.05000 | 2.0587e-03 | 8.7041e-14 | 2.005 |
|  41 | 0.02500 | 5.1420e-04 | 2.6112e-13 | 2.001 |
|  81 | 0.01250 | 1.2852e-04 | 1.3500e-12 | 2.000 |
| 161 | 0.00625 | 3.2128e-05 | 9.6918e-12 | 2.000 |
| 321 | 0.00313 | 8.0319e-06 | 2.1712e-11 | 2.000 |

- El **error L∞** (discretización) converge con orden 2.
- El **residuo** está cerca de la precisión de máquina: el sistema lineal se
  resuelve bien. Crece ligeramente al refinar porque los coeficientes de `A`
  son de tamaño `1/h^2` (redondeo, no un fallo).

El script muestra además 4 gráficas: solución numérica frente a exacta, error
puntual para varias mallas, convergencia log-log con referencia `O(h^2)` y
residuo frente a `h`.

### Ejemplo B · fuente constante y Dirichlet no homogénea

`-u'' = 1`, `u(0) = 0`, `u(1) = 1`. Solución analítica (integrando dos veces):

```text
u(x) = -x^2/2 + 3x/2 = x (3 - x) / 2
```

```powershell
python examples/poisson_dirichlet_b.py
```

|   N |       h | error L∞   | residuo    |
|----:|--------:|-----------:|-----------:|
|  11 | 0.10000 | 2.9843e-15 | 2.9843e-15 |
|  21 | 0.05000 | 1.4282e-14 | 5.6843e-14 |
|  41 | 0.02500 | 1.7506e-14 | 2.2737e-13 |
|  81 | 0.01250 | 1.0757e-13 | 1.8190e-12 |
| 161 | 0.00625 | 4.7717e-13 | 3.6380e-12 |
| 321 | 0.00313 | 1.0984e-12 | 1.4552e-11 |

- La solución es un polinomio de grado 2, así que el error de truncamiento
  `(h^2/12) u''''` es nulo: **el esquema es exacto**.
- El error que queda es solo redondeo y crece al refinar (peor
  condicionamiento), por lo que **no tiene sentido calcular un orden
  observado**.
- Este ejemplo sirve para comprobar que la condición no homogénea `u(1) = 1`
  se impone correctamente.

### Dependencias de los ejemplos

Las gráficas usan `matplotlib`:

```powershell
python -m pip install matplotlib
```

### Ramas y versiones

- Tags: `v0.1` (punto flotante), `v0.2` (operadores), `v0.3` (Poisson + gráficas).
- `feature/week3-fd-poisson`: rama de trabajo de v0.2–v0.3 (incluye los ejemplos A y B).
- `feature/ejemplo-clase`: ejercicio de clase (ejemplo B).