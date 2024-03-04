import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import rc

# Enable LaTeX rendering in Matplotlib
rc('text', usetex=True)

# Define the weight function
def weight_function(x):
    return np.sqrt(1 - x**2)

# Define the Chebyshev polynomials of the second kind
def chebyshev_second_kind(n, x):
    return np.polynomial.chebyshev.Chebyshev.basis(n)(x)

# Define the kernel function
def kernel_function(x, t):
    return x * t / (t - x)

# Define the right-hand side function
def f(x):
    return np.sqrt(1 - x**2) * chebyshev_second_kind(2, x) + 4 * x**5 - 3 * x**3

# Define the integrals in the system of equations
def A_kj(k, j):
    integrand = lambda x: weight_function(x) * chebyshev_second_kind(k, x) * chebyshev_second_kind(j, x)
    result, _ = quad(integrand, -1, 1)
    return result

def B_j(j):
    integrand = lambda x: chebyshev_second_kind(1, x)**2 * chebyshev_second_kind(j, x)
    result, _ = quad(integrand, -1, 1)
    return result / 2

def C_j(j):
    integrand = lambda x: chebyshev_second_kind(1, x) * chebyshev_second_kind(2, x) * chebyshev_second_kind(j, x)
    result, _ = quad(integrand, -1, 1)
    return result / 2

def D_j(j):
    integrand = lambda x: chebyshev_second_kind(1, x) * chebyshev_second_kind(3, x) * chebyshev_second_kind(j, x)
    result, _ = quad(integrand, -1, 1)
    return result / 2

def E_j(j):
    integrand = lambda x: chebyshev_second_kind(1, x) * chebyshev_second_kind(4, x) * chebyshev_second_kind(j, x)
    result, _ = quad(integrand, -1, 1)
    return result / 2

def F_j(j):
    integrand = lambda x: chebyshev_second_kind(1, x) * chebyshev_second_kind(5, x) * chebyshev_second_kind(j, x)
    result, _ = quad(integrand, -1, 1)
    return result / 2

def G_j(j):
    integrand = lambda x: f(x) * chebyshev_second_kind(j, x)
    result, _ = quad(integrand, -1, 1)
    return result

# Construct the linear system
A_matrix = np.array([[A_kj(k, j) for j in range(4)] for k in range(4)])
B_vector = np.array([B_j(j) for j in range(4)])
C_vector = np.array([C_j(j) for j in range(4)])
D_vector = np.array([D_j(j) for j in range(4)])
E_vector = np.array([E_j(j) for j in range(4)])
F_vector = np.array([F_j(j) for j in range(4)])
G_vector = np.array([G_j(j) for j in range(4)])

# Solve the linear system
solution = np.linalg.solve(A_matrix + B_vector[:, None] + C_vector[:, None] + D_vector[:, None] + E_vector[:, None] + F_vector[:, None], G_vector)

# Display the solution
print("Numerical solution coefficients:", solution)

# Construct the numerical solution function
def numerical_solution(x):
    return np.sum(solution[k] * chebyshev_second_kind(k, x) for k in range(4))

# Display the formula for the unknown function
def display_formula():
    formula = r"$u(x) = "
    for k in range(4):
        formula += f"{solution[k]:.4f}U_{k}(x) + "
    formula = formula[:-2] + "$"
    return formula

print("The unknow function is", display_formula())

# Display the numerical solution for a specific x
x_values = np.linspace(-1, 1, 1000)
numerical_results = [numerical_solution(x) for x in x_values]

# Plot the numerical solution
plt.plot(x_values, numerical_results, label='Numerical Solution')
plt.xlabel('x')
plt.ylabel(r'$u(x)$')
plt.title('Numerical Solution of the Singular Integral Equation')
plt.legend()

# Display the formula on the plot
plt.text(0.5, 2.5, display_formula(), fontsize=12, color='red')

# Show the plot
plt.show()