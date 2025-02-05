import pyomo.environ as pyo
from math import pi
import numpy as np

def cdf_formula(name):
    ''' Returns the CDF formula for the specified distribution. Currently, the following distributions are supported:
    - normal: normal distribution
    - sum-2-logistic-distributions: sum of two logistic distributions (see paper) 
    - sum-2-gaussian-distributions: sum of two gaussian distributions '''


    def cdf_normal(x, mu, sig, n):
        ''' Gaussian CDF computation via Abramowitz-Stegun approximation without if-statements (Pyomo cannot use If-Statements). '''

        z = (x - mu) / sig  # standardize the normal distribution

        epsilon = 1e-6  # to avoid division by zero => Needed for sign function approximation
        sign_z = z / (pyo.sqrt(z**2 + epsilon))  # sign function approximation

        z_abs = z * sign_z  # absolute value of z

        d1 = 0.0498673470  # Coefficients for the Abramowitz-Stegun approximation
        d2 = 0.0211410061
        d3 = 0.0032776263
        d4 = 0.0000380036
        d5 = 0.0000488906
        d6 = 0.0000053830 

        t = 1 + d1 * z_abs + d2 * z_abs**2 + d3 * z_abs**3 + d4 * z_abs**4 + d5 * z_abs**5 + d6 * z_abs**6 

        return 0.5 + 0.5 * sign_z * (1 - t**(-16))
    
    if name == 'normal':
        return cdf_normal
    elif name == 'sum-2-logistic-distributions':
        return lambda x, w1, w2, w3, w4, w5, w6: w1 / (1 + pyo.exp(-w2 *(x - w3))) + w4 / (1 + pyo.exp(-w5 *(x - w6)))
    elif name == 'sum-2-gaussian-distributions':
        return lambda x, w1, w2, mu1, mu2, sig1, sig2, n: w1 * cdf_normal(x, mu1, sig1, n) + w2 * cdf_normal(x, mu2, sig2, n)
    else:
        raise ValueError(f'CDF formula {name} not recognized')

def pdf_formula(name):
    ''' Returns the PDF formula for the specified distribution. '''
    if name == 'normal':
        return lambda x, mu, sig, n: 1 / (sig * pyo.sqrt(2 * pi)) * pyo.exp(-0.5 * ((x - mu) / sig)**2) 
    elif name == 'sum-2-logistic-distributions':
        return lambda x, w1, w2, w3, w4, w5, w6: w1 * w2 * pyo.exp(-w2 * (x - w3)) / (1 + pyo.exp(-w2 * (x - w3)))**2 + w4 * w5 * pyo.exp(-w5 * (x - w6)) / (1 + pyo.exp(-w5 * (x - w6)))**2
    elif name =='sum-2-gaussian-distributions':
        return lambda x, w1, w2, mu1, mu2, sig1, sig2, n: w1 / (sig1 * pyo.sqrt(2 * pi)) * pyo.exp(-0.5 * ((x - mu1) / sig1)**2) + w2 / (sig2 * pyo.sqrt(2 * pi)) * pyo.exp(-0.5 * ((x - mu2) / sig2)**2) 
    else:
        raise ValueError(f'PDF formula {name} not recognized')


def cdf_formula_numpy(name):
    ''' Returns the CDF formula for the specified distribution. 

    This is needed becasue during optimization, pyomo cannot utilize other libraries (like numpy). However, for the 
    plotting, pyomo cannot be used. Therefore, this function allows a computation of the CDF using numpy. '''


    def cdf_normal(x, mu, sig, n):
        ''' Abramowitz-Stegun approximation of the normal CDF '''
        z = (x - mu) / sig

        def phi(z):
            if z < 0: 
                return 1 - phi(-z)

            d1 = 0.0498673470
            d2 = 0.0211410061
            d3 = 0.0032776263
            d4 = 0.0000380036
            d5 = 0.0000488906
            d6 = 0.0000053830 

            t = 1 + d1 * z + d2 * z**2 + d3 * z**3 + d4 * z**4 + d5 * z**5 + d6 * z**6

            return 1 - 0.5 * (t ** -16)
            
        return phi(z)  

    if name == 'normal':
        return cdf_normal
    elif name == 'sum-2-gaussian-distributions':
        return lambda x, w1, w2, mu1, mu2, sig1, sig2, n: w1 * cdf_normal(x, mu1, sig1, n) + w2 * cdf_normal(x, mu2, sig2, n)
    elif name == 'sum-2-logistic-distributions':
        return lambda x, w1, w2, w3, w4, w5, w6: w1 / (1 + np.exp(-w2 *(x - w3))) + w4 / (1 + np.exp(-w5 *(x - w6)))
    else:
        raise ValueError(f'CDF formula {name} not recognized')



def dynamic_bounds(bounds, pdf_formula, weights):
    ''' Compute dynamic bounds to minimize errors when integrating the pdfs. Each time step has a different pdf shape 
    and therefore different bounds are needed for the integration.
    
    Args:
    bounds (tuple): lower and upper bound for the integration
    pdf_formula (function): function that computes the pdf
    weights (pd.DataFrame): weights for the pdf formula at each time step

    Returns:
    dynamic_bound_low (dict): lower bounds for the integration at each time step
    dynamic_bound_high (dict): upper bounds for the integration at each time step
    '''
    x = np.linspace(bounds[0], bounds[1], 10000)
    dynamic_bound_low = {}
    dynamic_bound_high = {}
    for t in weights.index:
        for i in range(len(x)):
            try:
                val = pdf_formula(x[i], *weights.loc[t])
                if val > 1e-8:
                    if i == 0:  # to avoid positive value at x[-1]
                        dynamic_bound_low[t] = x[i]
                    else:
                        dynamic_bound_low[t] = x[i-1]
                    break
            except OverflowError: # Happens if the argument inside the exponential function of the pdf is too large (or too small)
                continue

        for j in range(len(x)-2, 0, -1):
            try: 
                if pdf_formula(x[j], *weights.loc[t]) > 1e-8:
                    dynamic_bound_high[t] = x[j+1]
                    break
            except OverflowError:
                continue

    return dynamic_bound_low, dynamic_bound_high


def simpsons_rule(lb, ub, n, pdf, weights, offset=0):
    ''' Numerical integration of the pdf using Simpson's rule.

    If a better approximation of the integrals is necessary, a different approximation rule could be implemented. 
    See e.g. Gaussian quadrature that allows a variable step-length. 
    
    Args:
    lb (float): lower bound of the integration
    ub (float): upper bound of the integration
    n (int): number of breakpoints between lb and ub
    pdf (function): generic pdf formula
    weights (list): weights for the pdf formula
    offset (float): offset to represent a shifted pdf    
    '''
    h = (ub - lb) / n  # step size
    x = [lb + i * h for i in range(n+1)]
    integrand = lambda x, w: x * pdf((x + offset), *w)
    y = [integrand(xi, weights) for xi in x]

    approximated_integral = h / 3 * (y[0] + y[-1] + 4 * sum(y[1:-1:2]) + 2 * sum(y[2:-1:2]))
    return approximated_integral

