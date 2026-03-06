"""
AI Stats Lab
Random Variables and Distributions
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
# =========================================================
# QUESTION 1 — CDF Probabilities
# =========================================================
"""
    STEP 1
    Compute analytically

        P(X > 5)
        P(X < 5)
        P(3 < X < 7)

    STEP 2
    Simulate 100000 samples from Exp(1)

    STEP 3
    Estimate P(X > 5) using simulation

    RETURN

        analytic_gt5
        analytic_lt5
        analytic_interval
        simulated_gt5
"""
def cdf_probabilities():

    # Analytical calculations
    analytic_gt5 = math.exp(-5)

    analytic_lt5 = 1 - math.exp(-5)

    analytic_interval = math.exp(-3) - math.exp(-7)

    # Monte Carlo Simulation
    samples = np.random.exponential(1, 100000)

    simulated_gt5 = np.mean(samples > 5)

    return analytic_gt5, analytic_lt5, analytic_interval, simulated_gt5

# =========================================================
# QUESTION 2 — PDF Validation and Plot
# =========================================================

def pdf_validation_plot():
    """
    Candidate PDF

        f(x) = 2x e^{-x^2} for x >= 0

    STEP 1
    Verify non-negativity

    STEP 2
    Compute

        integral_0^∞ f(x) dx

    STEP 3
    Determine if valid PDF

    STEP 4
    Plot f(x) on [0,3]

    RETURN

        integral_value
        is_valid_pdf
    """
    def f(x):
        return 2 * x * np.exp(-x**2)

    # compute integral
    integral_value, _ = quad(f, 0, np.inf)

    # check if valid pdf
    if abs(integral_value - 1) < 0.001:
        is_valid_pdf = True
    else:
        is_valid_pdf = False

    # plot the function
    x = np.linspace(0, 3, 200)
    y = f(x)

    plt.plot(x, y)
    plt.title("PDF: 2x e^(-x^2)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()

    return integral_value, is_valid_pdf
# =========================================================
# QUESTION 3 — Exponential Distribution
# =========================================================
"""
    X ~ Exp(1)

    STEP 1
    Compute analytically

        P(X > 5)
        P(1 < X < 3)

    STEP 2
    Simulate 100000 samples

    STEP 3
    Estimate probabilities using simulation

    RETURN

        analytic_gt5
        analytic_interval
        simulated_gt5
        simulated_interval
    """
def exponential_probabilities():
    # Analytical
    analytic_gt5 = math.exp(-5)

    analytic_interval = math.exp(-1) - math.exp(-3)

    # Simulation
    samples = np.random.exponential(1, 100000)

    simulated_gt5 = np.mean(samples > 5)

    simulated_interval = np.mean((samples > 1) & (samples < 3))

    return analytic_gt5, analytic_interval, simulated_gt5, simulated_interval

# =========================================================
# QUESTION 4 — Gaussian Distribution
# =========================================================
"""
    X ~ N(10,2^2)

    STEP 1
    Standardize variable

        Z = (X - 10)/2

    STEP 2
    Compute analytically

        P(X ≤ 12)
        P(8 < X < 12)

    STEP 3
    Simulate 100000 samples

    STEP 4
    Estimate probabilities

    RETURN

        analytic_le12
        analytic_interval
        simulated_le12
        simulated_interval
    """
def gaussian_probabilities():

    mu = 10
    sigma = 2

    # Analytical
    analytic_le12 = norm.cdf(12, mu, sigma)

    analytic_interval = norm.cdf(12, mu, sigma) - norm.cdf(8, mu, sigma)

    # Simulation
    samples = np.random.normal(mu, sigma, 100000)

    simulated_le12 = np.mean(samples <= 12)

    simulated_interval = np.mean((samples > 8) & (samples < 12))

    return analytic_le12, analytic_interval, simulated_le12, simulated_interval
