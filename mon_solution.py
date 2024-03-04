import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import rc

# Enable LaTeX rendering in Matplotlib
rc('text', usetex=True)

def weight_function(x):
    return np.sqrt(1 - x**2)

def chebyshev_second_kind(n, x):
    return np.polynomial.chebyshev.Chebyshev.basis(n)(x)

def kernel_function(x, t):
    return x * t / (t - x)

def f(x):
    return np.sqrt(1 - x**2) * chebyshev_second_kind(2, x) + 4 * x**5 - 3 * x**3

def integral_terms(k, j):
    integrand = lambda x: weight_function(x) * chebyshev_second_kind(k, x) * chebyshev_second_kind(j, x)
    result, _ = quad(integrand, -1, 1)
    return result

def construct_linear_system():
    A_matrix = np.array([[integral_terms(k, j) for j in range(4)] for k in range(4)])
    B_vector = np.array([quad(lambda x: chebyshev_second_kind(1, x)**2 * chebyshev_second_kind(j, x), -1, 1)[0] / 2 for j in range(4)])
    C_vector = np.array([quad(lambda x: chebyshev_second_kind(1, x) * chebyshev_second_kind(2, x) * chebyshev_second_kind(j, x), -1, 1)[0] / 2 for j in range(4)])
    D_vector = np.array([quad(lambda x: chebyshev_second_kind(1, x) * chebyshev_second_kind(3, x) * chebyshev_second_kind(j, x), -1, 1)[0] / 2 for j in range(4)])
    E_vector = np.array([quad(lambda x: chebyshev_second_kind(1, x) * chebyshev_second_kind(4, x) * chebyshev_second_kind(j, x), -1, 1)[0] / 2 for j in range(4)])
    F_vector = np.array([quad(lambda x: chebyshev_second_kind(1, x) * chebyshev_second_kind(5, x) * chebyshev_second_kind(j, x), -1, 1)[0] / 2 for j in range(4)])
    G_vector = np.array([quad(lambda x: f(x) * chebyshev_second_kind(j, x), -1, 1)[0] for j in range(4)])

    return A_matrix + B_vector[:, None] + C_vector[:, None] + D_vector[:, None] + E_vector[:, None] + F_vector[:, None], G_vector

def solve_linear_system():
    A, b = construct_linear_system()
    return np.linalg.solve(A, b)

def numerical_solution(x, coefficients):
    return np.sum(coefficients[k] * chebyshev_second_kind(k, x) for k in range(4))

def display_formula(coefficients):
    formula = r"$u(x) = "
    for k in range(4):
        formula += f"{coefficients[k]:.4f}U_{k}(x) + "
    formula = formula[:-2] + "$"
    return formula

# Solve the linear system
solution_coefficients = solve_linear_system()

# Display the solution coefficients
print("Numerical solution coefficients:", solution_coefficients)

# Display the formula for the unknown function
print(display_formula(solution_coefficients))

# Construct the numerical solution function
def numerical_solution_function(x):
    return numerical_solution(x, solution_coefficients)

# Display the numerical solution for a specific x
x_values = np.linspace(-1, 1, 1000)
numerical_results = [numerical_solution_function(x) for x in x_values]

# Plot the numerical solution
plt.plot(x_values, numerical_results, label='Numerical Solution')
plt.xlabel('x')
plt.ylabel(r'$u(x)$')
plt.title('Numerical Solution of the Singular Integral Equation')
plt.legend()

# Display the formula on the plot
plt.text(0.5, 2.5, display_formula(solution_coefficients), fontsize=12, color='red')

# Show the plot
plt.show()
