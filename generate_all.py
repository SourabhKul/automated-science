import os

DOMAINS = [
    {
        "domain": "ecology",
        "desc": "Predator-Prey Dynamics (Hudson Bay Lynx-Hare)",
        "state_vars": "2 elements: [H (Hare), L (Lynx)]",
        "time_req": "100 yearly timepoints (years 0-100)",
        "hints": "- The real predator-prey dynamics exhibit Allee effects and spatial constraints\n- Consider Holling Type II or Type III functional responses replacing the linear `a*H*L` term\n- Carrying capacity on the prey (logistic growth instead of exponential)",
        "y0": "[40.0, 9.0]",
        "params": "[1.0, 0.1, 0.05, 0.5]", # r, a, b, m
        "seed": "def dynamics(t, y, args):\n    r, a, b, m = args\n    H, L = y\n    dH = r * H - a * H * L\n    dL = b * H * L - m * L\n    return jnp.array([dH, dL])\n\nmetadata = [\n    {'name': 'r', 'range': (0.1, 2.0)},\n    {'name': 'a', 'range': (0.01, 0.5)},\n    {'name': 'b', 'range': (0.01, 0.5)},\n    {'name': 'm', 'range': (0.1, 2.0)}\n]"
    },
    {
        "domain": "bz_chem",
        "desc": "Belousov-Zhabotinsky (BZ) Chemical Oscillator",
        "state_vars": "3 elements: [X (HBrO2), Y (Br-), Z (Ce4+)]",
        "time_req": "200 timepoints over 50 seconds",
        "hints": "- The fundamental reaction exhibits non-linear Oregonator kinetics\n- Look into autocatalytic steps with quadratic dependencies (X^2 terms)\n- Consider modifying the stoichiometric factor f",
        "y0": "[0.1, 0.1, 0.1]",
        "params": "[0.2, 0.1, 1.0, 0.5, 0.2]", # q, f, eps, alpha, beta
        "seed": "def dynamics(t, y, args):\n    q, f, eps, alpha, beta = args\n    X, Y, Z = y\n    dX = (q*Y - X*Y + X*(1-X)) / eps\n    dY = (-q*Y - X*Y + f*Z) / alpha\n    dZ = (X - Z) / beta\n    return jnp.array([dX, dY, dZ])\n\nmetadata = [\n    {'name': 'q', 'range': (0.01, 0.5)},\n    {'name': 'f', 'range': (0.1, 2.0)},\n    {'name': 'eps', 'range': (0.01, 2.0)},\n    {'name': 'alpha','range': (0.1, 2.0)},\n    {'name': 'beta', 'range': (0.1, 2.0)}\n]"
    },
    {
        "domain": "oncology",
        "desc": "Tumor Growth and Angiogenesis",
        "state_vars": "2 elements: [V (Tumor Volume), K (Carrying Capacity)]",
        "time_req": "100 timepoints over 50 days",
        "hints": "- Hahnfeldt models incorporate endothelial cell proliferation and baseline vessel death\n- Implement Gompertzian growth rather than logistic \n- K grows via a stimulatory term proportional to V and decays via an inhibitory term",
        "y0": "[10.0, 20.0]",
        "params": "[0.1, 0.05, 0.02]", # alpha, b, d
        "seed": "def dynamics(t, y, args):\n    alpha, b, d = args\n    V, K = y\n    dV = alpha * V * jnp.log(K / V)\n    dK = b * V - d * K * (V ** (2/3))\n    return jnp.array([dV, dK])\n\nmetadata = [\n    {'name': 'alpha', 'range': (0.01, 0.5)},\n    {'name': 'b', 'range': (0.01, 0.5)},\n    {'name': 'd', 'range': (0.01, 0.5)}\n]"
    },
    {
        "domain": "pkpd",
        "desc": "Propofol Pharmacokinetics (3-Compartment)",
        "state_vars": "3 elements: [C1 (Central), C2 (Fast Periph), C3 (Slow Periph)]",
        "time_req": "120 timepoints over 120 minutes",
        "hints": "- Include non-linear drug clearance mechanisms\n- Look into age-dependent or weight-adjusted covariates modifying inter-compartmental clearance (Q2, Q3)\n- Add infusion inputs as a forcing function if possible",
        "y0": "[10.0, 0.0, 0.0]",
        "params": "[0.1, 0.05, 0.02, 0.05, 0.01]", # k10, k12, k21, k13, k31
        "seed": "def dynamics(t, y, args):\n    k10, k12, k21, k13, k31 = args\n    C1, C2, C3 = y\n    dC1 = -(k10 + k12 + k13)*C1 + k21*C2 + k31*C3\n    dC2 = k12*C1 - k21*C2\n    dC3 = k13*C1 - k31*C3\n    return jnp.array([dC1, dC2, dC3])\n\nmetadata = [\n    {'name': 'k10', 'range': (0.01, 0.5)},\n    {'name': 'k12', 'range': (0.01, 0.5)},\n    {'name': 'k21', 'range': (0.01, 0.5)},\n    {'name': 'k13', 'range': (0.01, 0.5)},\n    {'name': 'k31', 'range': (0.01, 0.5)}\n]"
    },
    {
        "domain": "cardio",
        "desc": "Cardiovascular Hemodynamics (Windkessel)",
        "state_vars": "2 elements: [P_ao (Aortic Pressure), Q (Flow)]",
        "time_req": "200 timepoints over 2 seconds (high frequency)",
        "hints": "- 2-element Windkessel misses characteristic impedance; try adding it\n- Introduce non-linear aortic compliance C(P)\n- Fluid momentum (inertance L) is crucial for accurate dQ/dt",
        "y0": "[80.0, 0.0]",
        "params": "[1.0, 0.05, 0.1]", # R_p, C, R_c
        "seed": "def dynamics(t, y, args):\n    R_p, C, R_c = args\n    P, Q = y\n    # Simplified mock forced flow\n    flow_in = 100 * jnp.sin(t * jnp.pi)\n    dP = (flow_in - P/R_p) / C\n    dQ = P / R_c - Q * 2.0\n    return jnp.array([dP, dQ])\n\nmetadata = [\n    {'name': 'R_p', 'range': (0.5, 3.0)},\n    {'name': 'C', 'range': (0.01, 0.2)},\n    {'name': 'R_c', 'range': (0.01, 0.5)}\n]"
    },
    {
        "domain": "battery",
        "desc": "Lithium-ion Battery Capacity Fade",
        "state_vars": "2 elements: [Q (Capacity), S (SEI Layer Thickness)]",
        "time_req": "500 timepoints representing cycles 0 to 500",
        "hints": "- SEI layer growth is proportional to sqrt(time) or cycle number\n- Implement stress-induced active material loss\n- Couple volume expansion to SEI cracking mechanisms",
        "y0": "[100.0, 1.0]",
        "params": "[0.01, 0.5, 0.05]", # k_sei, n, k_aml
        "seed": "def dynamics(t, y, args):\n    k_sei, n, k_aml = args\n    Q, S = y\n    dS = k_sei * (t ** n)\n    dQ = -k_aml * Q - 0.1 * dS\n    return jnp.array([dQ, dS])\n\nmetadata = [\n    {'name': 'k_sei', 'range': (0.001, 0.1)},\n    {'name': 'n', 'range': (0.1, 0.9)},\n    {'name': 'k_aml', 'range': (0.01, 0.1)}\n]"
    },
    {
        "domain": "synbio",
        "desc": "Synthetic Biology Genetic Toggle Switch",
        "state_vars": "2 elements: [U (Repressor 1), V (Repressor 2)]",
        "time_req": "100 timepoints over 50 hours",
        "hints": "- Classic Gardner model uses mutually repressive Hill kinetics\n- Consider leaky transcription (basal promoter activity)\n- Consider cooperative binding (higher Hill coefficients) or inducer dynamics",
        "y0": "[2.0, 0.1]",
        "params": "[2.0, 2.0, 2.0, 2.0]", # a1, a2, gamma, n
        "seed": "def dynamics(t, y, args):\n    a1, a2, gamma, n = args\n    U, V = y\n    dU = a1 / (1 + V**n) - gamma * U\n    dV = a2 / (1 + U**n) - gamma * V\n    return jnp.array([dU, dV])\n\nmetadata = [\n    {'name': 'a1', 'range': (0.5, 5.0)},\n    {'name': 'a2', 'range': (0.5, 5.0)},\n    {'name': 'gamma', 'range': (0.5, 5.0)},\n    {'name': 'n', 'range': (1.0, 5.0)}\n]"
    },
    {
        "domain": "econ",
        "desc": "Goodwin Growth Cycle (Macroeconomics)",
        "state_vars": "2 elements: [v (Employment rate), u (Wage share)]",
        "time_req": "200 timepoints over 50 years",
        "hints": "- Integrate a Phillips curve with non-linear inflation expectations\n- Incorporate time-delays in capital investment\n- Couple financial debt mechanisms (Minsky financial instability hypothesis)",
        "y0": "[0.8, 0.6]",
        "params": "[0.02, 0.05, 0.1, 0.1]", # alpha, beta, gamma, rho
        "seed": "def dynamics(t, y, args):\n    alpha, beta, gamma, rho = args\n    v, u = y\n    dv = v * (1 - u) - alpha * v\n    du = u * (rho * v - gamma) - beta * u\n    return jnp.array([dv, du])\n\nmetadata = [\n    {'name': 'alpha', 'range': (0.01, 0.1)},\n    {'name': 'beta', 'range': (0.01, 0.2)},\n    {'name': 'gamma', 'range': (0.05, 0.5)},\n    {'name': 'rho', 'range': (0.05, 0.5)}\n]"
    },
    {
        "domain": "epidemiology",
        "desc": "Multi-Strain Pathogen Competition",
        "state_vars": "4 elements: [S (Susceptible), I1 (Strain 1), I2 (Strain 2), R (Recovered)]",
        "time_req": "150 timepoints over 150 days",
        "hints": "- Waning immunity (R -> S transitions)\n- Cross-immunity coefficients (infection to I1 limits susceptibility to I2)\n- Behavioral feedback (transmission rates drop as cases rise)",
        "y0": "[0.9, 0.05, 0.05, 0.0]",
        "params": "[0.4, 0.6, 0.1, 0.1]", # beta1, beta2, gamma1, gamma2
        "seed": "def dynamics(t, y, args):\n    beta1, beta2, gamma1, gamma2 = args\n    S, I1, I2, R = y\n    dS = -beta1 * S * I1 - beta2 * S * I2\n    dI1 = beta1 * S * I1 - gamma1 * I1\n    dI2 = beta2 * S * I2 - gamma2 * I2\n    dR = gamma1 * I1 + gamma2 * I2\n    return jnp.array([dS, dI1, dI2, dR])\n\nmetadata = [\n    {'name': 'beta1', 'range': (0.1, 1.0)},\n    {'name': 'beta2', 'range': (0.1, 1.0)},\n    {'name': 'gamma1', 'range': (0.05, 0.3)},\n    {'name': 'gamma2', 'range': (0.05, 0.3)}\n]"
    },
    {
        "domain": "climate",
        "desc": "Glacial Cycles (Saltzman)",
        "state_vars": "3 elements: [I (Ice Mass), mu (CO2), theta (Ocean Temp)]",
        "time_req": "400 timepoints over 400 kyr",
        "hints": "- Sea ice albedo feedback introduces extreme non-linearities\n- Deep ocean carbon sequestration (CO2 lag behind temperature)\n- Add astronomical Milankovitch forcing as an explicit sinusoidal input",
        "y0": "[1.0, 1.0, 1.0]",
        "params": "[0.1, 0.2, 0.1, 0.5, 0.2]", # a, b, c, d, e
        "seed": "def dynamics(t, y, args):\n    a, b, c, d, e = args\n    I, mu, theta = y\n    dI = -I + a * mu + b * theta\n    dmu = c * I - d * mu\n    dtheta = -e * I + 0.1 * mu\n    return jnp.array([dI, dmu, dtheta])\n\nmetadata = [\n    {'name': 'a', 'range': (0.01, 1.0)},\n    {'name': 'b', 'range': (0.01, 1.0)},\n    {'name': 'c', 'range': (0.01, 1.0)},\n    {'name': 'd', 'range': (0.01, 1.0)},\n    {'name': 'e', 'range': (0.01, 1.0)}\n]"
    },
    {
        "domain": "neuro",
        "desc": "FitzHugh-Nagumo Neuronal Spiking",
        "state_vars": "2 elements: [V (Voltage), W (Recovery)]",
        "time_req": "200 timepoints over 200 ms",
        "hints": "- The voltage V exhibits fast cubic nonlinearity.\n- Recovery W is a slow linear variable.",
        "y0": "[-1.0, 0.0]",
        "params": "[0.7, 0.8, 0.08, 0.5]",
        "seed": "def dynamics(t, y, args):\n    a, b, tau, I_ext = args\n    V, W = y\n    dV = V - (V**3)/3.0 - W + I_ext\n    dW = (V + a - b * W) / tau\n    return jnp.array([dV, dW])\n\nmetadata = [\n    {'name': 'a', 'range': (0.1, 1.0)},\n    {'name': 'b', 'range': (0.1, 1.0)},\n    {'name': 'tau', 'range': (0.01, 0.2)},\n    {'name': 'I_ext', 'range': (0.1, 1.0)}\n]"
    },
    {
        "domain": "fluid",
        "desc": "Lorenz Attractor (Convection)",
        "state_vars": "3 elements: [X (Convection), Y (Horiz Temp), Z (Vert Temp)]",
        "time_req": "400 timepoints over 40 seconds",
        "hints": "- The system is highly chaotic and sensitive to initial conditions.\n- X and Y cross-couple non-linearly.",
        "y0": "[1.0, 1.0, 1.0]",
        "params": "[10.0, 28.0, 2.66]",
        "seed": "def dynamics(t, y, args):\n    sigma, rho, beta = args\n    X, Y, Z = y\n    dX = sigma * (Y - X)\n    dY = X * (rho - Z) - Y\n    dZ = X * Y - beta * Z\n    return jnp.array([dX, dY, dZ])\n\nmetadata = [\n    {'name': 'sigma', 'range': (5.0, 15.0)},\n    {'name': 'rho', 'range': (15.0, 35.0)},\n    {'name': 'beta', 'range': (1.0, 5.0)}\n]"
    },
    {
        "domain": "astro",
        "desc": "Orbital Mechanics (Two-Body)",
        "state_vars": "4 elements: [x, y, vx, vy]",
        "time_req": "500 timepoints over 50 years",
        "hints": "- Gravity follows an inverse-square law based on distance r.\n- r = sqrt(x^2 + y^2).",
        "y0": "[1.0, 0.0, 0.0, 1.0]",
        "params": "[1.0]",
        "seed": "def dynamics(t, y, args):\n    GM = args[0]\n    x, y, vx, vy = y\n    r3 = jnp.maximum((x**2 + y**2)**1.5, 1e-6)\n    dx = vx\n    dy = vy\n    dvx = -GM * x / r3\n    dvy = -GM * y / r3\n    return jnp.array([dx, dy, dvx, dvy])\n\nmetadata = [\n    {'name': 'GM', 'range': (0.1, 5.0)}\n]"
    },
    {
        "domain": "chem_kinetics",
        "desc": "Michaelis-Menten Enzyme Kinetics",
        "state_vars": "4 elements: [S (Substrate), E (Enzyme), ES (Complex), P (Product)]",
        "time_req": "100 timepoints over 10 seconds",
        "hints": "- Total enzyme E + ES is conserved.\n- Substrate binds reversibly to form ES, which decays to P.",
        "y0": "[10.0, 1.0, 0.0, 0.0]",
        "params": "[1.0, 0.5, 0.2]",
        "seed": "def dynamics(t, y, args):\n    k1, k_1, k2 = args\n    S, E, ES, P = y\n    dS = -k1 * S * E + k_1 * ES\n    dE = -k1 * S * E + (k_1 + k2) * ES\n    dES = k1 * S * E - (k_1 + k2) * ES\n    dP = k2 * ES\n    return jnp.array([dS, dE, dES, dP])\n\nmetadata = [\n    {'name': 'k1', 'range': (0.1, 2.0)},\n    {'name': 'k_1', 'range': (0.01, 1.0)},\n    {'name': 'k2', 'range': (0.01, 1.0)}\n]"
    },
    {
        "domain": "immune",
        "desc": "T-Cell Pathogen Dynamics",
        "state_vars": "2 elements: [P (Pathogen), T (T-cells)]",
        "time_req": "150 timepoints over 15 days",
        "hints": "- Pathogens replicate exponentially until countered by T-cells.\n- T-cells expand proportionally to pathogen load.",
        "y0": "[1.0, 0.1]",
        "params": "[0.5, 0.1, 0.2, 0.05]",
        "seed": "def dynamics(t, y, args):\n    r, a, c, d = args\n    P, T = y\n    dP = r * P - a * P * T\n    dT = c * P * T - d * T\n    return jnp.array([dP, dT])\n\nmetadata = [\n    {'name': 'r', 'range': (0.1, 1.0)},\n    {'name': 'a', 'range': (0.01, 0.5)},\n    {'name': 'c', 'range': (0.01, 0.5)},\n    {'name': 'd', 'range': (0.01, 0.2)}\n]"
    },
    {
        "domain": "pop_genetics",
        "desc": "Allele Frequency Drift (Selection-Mutation)",
        "state_vars": "1 element: [p (Allele frequency)]",
        "time_req": "200 timepoints over 200 generations",
        "hints": "- Frequency p must stay bounded between 0 and 1.\n- Selection introduces a p*(1-p) term.",
        "y0": "[0.5]",
        "params": "[0.05, 0.01, 0.01]",
        "seed": "def dynamics(t, y, args):\n    s, mu, nu = args\n    p = y[0]\n    dp = s * p * (1.0 - p) - mu * p + nu * (1.0 - p)\n    return jnp.array([dp])\n\nmetadata = [\n    {'name': 's', 'range': (0.01, 0.2)},\n    {'name': 'mu', 'range': (0.001, 0.05)},\n    {'name': 'nu', 'range': (0.001, 0.05)}\n]"
    },
    {
        "domain": "thermo",
        "desc": "Newton Cooling with Non-Linear Radiation",
        "state_vars": "1 element: [T (Temperature)]",
        "time_req": "100 timepoints over 100 minutes",
        "hints": "- Linear conduction cooling coupled with Stefan-Boltzmann T^4 radiation cooling.",
        "y0": "[500.0]",
        "params": "[0.1, 1e-9, 300.0]",
        "seed": "def dynamics(t, y, args):\n    k, eps_sig, T_env = args\n    T = y[0]\n    dT = -k * (T - T_env) - eps_sig * (T**4 - T_env**4)\n    return jnp.array([dT])\n\nmetadata = [\n    {'name': 'k', 'range': (0.01, 0.5)},\n    {'name': 'eps_sig', 'range': (1e-10, 1e-8)},\n    {'name': 'T_env', 'range': (250.0, 350.0)}\n]"
    },
    {
        "domain": "materials",
        "desc": "Fatigue Crack Growth (Paris Law)",
        "state_vars": "1 element: [a (Crack Length)]",
        "time_req": "300 timepoints over 300 cycles",
        "hints": "- Crack length grows with stress intensity factor raised to power m.\n- Typically da/dt = C * (dK * sqrt(a))^m.",
        "y0": "[0.01]",
        "params": "[1e-6, 3.0, 50.0]",
        "seed": "def dynamics(t, y, args):\n    C, m, dK = args\n    a = jnp.maximum(y[0], 1e-6)\n    da = C * (dK * jnp.sqrt(a))**m\n    return jnp.array([da])\n\nmetadata = [\n    {'name': 'C', 'range': (1e-8, 1e-4)},\n    {'name': 'm', 'range': (2.0, 5.0)},\n    {'name': 'dK', 'range': (10.0, 100.0)}\n]"
    },
    {
        "domain": "agriculture",
        "desc": "Crop Growth and Soil Depletion",
        "state_vars": "2 elements: [B (Biomass), N (Nutrients)]",
        "time_req": "100 timepoints over 100 days",
        "hints": "- Biomass grows proportionally to available nutrients.\n- Nutrients deplete as biomass increases.",
        "y0": "[0.1, 10.0]",
        "params": "[0.5, 0.1, 0.05]",
        "seed": "def dynamics(t, y, args):\n    r, alpha, m = args\n    B, N = y\n    dB = r * B * N - m * B\n    dN = -alpha * r * B * N\n    return jnp.array([dB, dN])\n\nmetadata = [\n    {'name': 'r', 'range': (0.1, 1.0)},\n    {'name': 'alpha', 'range': (0.01, 0.5)},\n    {'name': 'm', 'range': (0.01, 0.2)}\n]"
    },
    {
        "domain": "social",
        "desc": "Bass Diffusion (Idea/Product Spread)",
        "state_vars": "1 element: [A (Adopters)]",
        "time_req": "50 timepoints over 50 months",
        "hints": "- Innovation (p) and Imitation (q) drive adoption.\n- Adoption is capped at Market Size M.",
        "y0": "[0.01]",
        "params": "[0.03, 0.38, 1.0]",
        "seed": "def dynamics(t, y, args):\n    p, q, M = args\n    A = y[0]\n    dA = (p + q * (A / M)) * (M - A)\n    return jnp.array([dA])\n\nmetadata = [\n    {'name': 'p', 'range': (0.01, 0.1)},\n    {'name': 'q', 'range': (0.1, 0.9)},\n    {'name': 'M', 'range': (0.5, 2.0)}\n]"
    },
    {
        "domain": "real_sunspots",
        "desc": "Real Sunspot Activity (1700-2008)",
        "state_vars": "1 element: [S (Sunspots)]",
        "time_req": "309 yearly timepoints",
        "hints": "- Data is highly oscillatory but noisy. Consider an autoregressive-like continuous ODE or a driven oscillator.",
        "y0": "[5.0]",
        "params": "[0.1, 10.0, 0.05]",
        "seed": "def dynamics(t, y, args):\n    omega, A, d = args\n    S = y[0]\n    dS = A * jnp.sin(omega * t) - d * S\n    return jnp.array([dS])\n\nmetadata = [\n    {'name': 'omega', 'range': (0.01, 1.0)},\n    {'name': 'A', 'range': (1.0, 50.0)},\n    {'name': 'd', 'range': (0.01, 0.5)}\n]",
        "is_real": True
    },
    {
        "domain": "real_nile",
        "desc": "Real Nile River Flow (1871-1970)",
        "state_vars": "1 element: [V (Volume)]",
        "time_req": "100 yearly timepoints",
        "hints": "- Mean-reverting stochastic-like process. Try Ornstein-Uhlenbeck style deterministic drift.",
        "y0": "[1120.0]",
        "params": "[0.1, 1000.0]",
        "seed": "def dynamics(t, y, args):\n    k, V_mean = args\n    V = y[0]\n    dV = k * (V_mean - V)\n    return jnp.array([dV])\n\nmetadata = [\n    {'name': 'k', 'range': (0.01, 1.0)},\n    {'name': 'V_mean', 'range': (800.0, 1200.0)}\n]",
        "is_real": True
    },
    {
        "domain": "real_macro",
        "desc": "Real US GDP (1959-2009)",
        "state_vars": "1 element: [G (GDP)]",
        "time_req": "203 quarterly timepoints",
        "hints": "- Exponential growth with periodic recessionary shocks.",
        "y0": "[2710.0]",
        "params": "[0.03]",
        "seed": "def dynamics(t, y, args):\n    g = args[0]\n    G = y[0]\n    dG = g * G\n    return jnp.array([dG])\n\nmetadata = [\n    {'name': 'g', 'range': (0.001, 0.1)}\n]",
        "is_real": True
    },
    {
        "domain": "real_theophylline",
        "desc": "Real Theophylline PK (Subject 1)",
        "state_vars": "1 element: [C (Concentration)]",
        "time_req": "11 timepoints over 24 hours",
        "hints": "- Classic 1-compartment oral absorption model. C rises fast, decays exponentially.",
        "y0": "[0.74]",
        "params": "[1.5, 0.1]",
        "seed": "def dynamics(t, y, args):\n    ka, ke = args\n    C = y[0]\n    dC = ka * 4.02 * jnp.exp(-ka * t) - ke * C\n    return jnp.array([dC])\n\nmetadata = [\n    {'name': 'ka', 'range': (0.1, 5.0)},\n    {'name': 'ke', 'range': (0.01, 1.0)}\n]",
        "is_real": True
    },
    {
        "domain": "real_co2",
        "desc": "Real Mauna Loa CO2 (1958-2001)",
        "state_vars": "1 element: [C (CO2)]",
        "time_req": "43 yearly timepoints",
        "hints": "- Accelerating exponential growth.",
        "y0": "[315.0]",
        "params": "[0.01, 1.0]",
        "seed": "def dynamics(t, y, args):\n    r, a = args\n    C = y[0]\n    dC = r * (C - 280.0) + a\n    return jnp.array([dC])\n\nmetadata = [\n    {'name': 'r', 'range': (0.001, 0.1)},\n    {'name': 'a', 'range': (0.1, 5.0)}\n]",
        "is_real": True
    }
]

ORCHESTRATOR_TEMPLATE = '''import os
import sys
import numpy as np
import jax.numpy as jnp
import requests
import re
import csv
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from automated_science.core.sbi_engine import SBIEngine
except:
    from core.sbi_engine import SBIEngine

LLM_ENDPOINT   = "http://localhost:1234/v1/chat/completions"
MODEL_ID       = "qwen3.6-27b-nvfp4"
EXPERIMENT_TAG = "qwen36_27b_{domain}"

MODELS_DIR      = f"models/{{EXPERIMENT_TAG}}"
LOG_FILE        = f"{{EXPERIMENT_TAG}}_run_history.csv"
BEST_LOGIC_PATH = f"{{MODELS_DIR}}/best_logic.py"
RUN_LOG_PATH    = f"{{EXPERIMENT_TAG}}_run.log"

os.makedirs(MODELS_DIR, exist_ok=True)

class _Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()

_log_fh = open(RUN_LOG_PATH, "a", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_fh)
sys.stderr = _Tee(sys.__stderr__, _log_fh)

def call_llm(prompt, system_prompt="You are a mathematical AI. Output ONLY a Python code block with `dynamics(t, y, args)` and `metadata`. No prose."):
    payload = {{
        "model": MODEL_ID, 
        "messages": [
            {{"role": "system", "content": system_prompt}}, 
            {{"role": "user", "content": prompt}}
        ], 
        "temperature": 0.2, 
        "max_tokens": 8192
    }}
    try:
        r = requests.post(LLM_ENDPOINT, json=payload, timeout=900)
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"LLM Error: {{e}}")
        return None

def extract_code(llm_response):
    if not llm_response: return None, None
    m = re.search(r"<think>(.*?)</think>", llm_response, re.DOTALL)
    thoughts = m.group(1).strip() if m else ""
    match = re.search(r"```python(.*?)```", llm_response, re.DOTALL)
    if match: return match.group(1).strip(), thoughts
    fallback = re.search(r"(def dynamics\(.*)", llm_response, re.DOTALL)
    if fallback: return fallback.group(1).strip(), thoughts
    return re.sub(r"^```[a-z]*\\n?", "", llm_response.strip()).replace("```", "").strip(), thoughts

def run_loop():
    print("=" * 70)
    print(f"  QWEN-3.6 27B · {desc}")
    print(f"  Domain      : {domain}")
    print("=" * 70)

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["Timestamp", "Iteration", "Action", "Proposed_Loss", "Baseline_Loss"])

    def log_action(iteration, action, prop_loss, base_loss):
        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), iteration, action, prop_loss, base_loss])

    gt_data  = np.load("data/{domain}_ground_truth.npy")
    t_points = np.load("data/{domain}_time_points.npy")
    engine   = SBIEngine(gt_data, t_points)

    def evaluate_llm_logic(logic_code_string):
        full_module = f"""import jax.numpy as jnp\\nfrom core.base_model import BaseModel\\nfrom diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5\\n\\n{{logic_code_string}}\\n\\nclass CandidateModel(BaseModel):\\n    def simulate(self, params, time_points, y0):\\n        saveat = SaveAt(ts=time_points)\\n        try:\\n            sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0=float(time_points[0]), t1=float(time_points[-1]), dt0=0.5, y0=y0, args=params, saveat=saveat, max_steps=10000)\\n            return sol.ys\\n        except Exception:\\n            return jnp.zeros((len(time_points), y0.shape[0]))\\n    def get_parameter_metadata(self):\\n        return metadata\\n    def get_initial_conditions(self):\\n        return jnp.array({y0})\\n    def get_latex(self):\\n        return \\"Qwen-3.6 Generated {domain}\\"\\n"""
        import importlib.util
        module_name = f"dynamic_{domain}_{{np.random.randint(100000)}}"
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        mod  = importlib.util.module_from_spec(spec)
        try:
            exec(full_module, mod.__dict__)
            model = mod.CandidateModel()
            # Safety check: Can the model simulate once without crashing?
            # We use a dummy parameter set from the metadata
            test_params = jnp.array([p['range'][0] for p in model.get_parameter_metadata()])
            _ys = model.simulate(test_params, t_points[:5], model.get_initial_conditions())
            if jnp.any(jnp.isnan(_ys)) or jnp.any(jnp.isinf(_ys)):
                raise ValueError("Model produced NaN/Inf in test simulation.")
            
            res = engine.run_abc_smc(model, target_samples=100, generations=10, initial_particles=10000)
            return res["median_distance"], full_module, None
        except Exception as e:
            return float("inf"), full_module, str(e)

    current_logic = """{seed}"""

    best_loss, full_module, _err = evaluate_llm_logic(current_logic)
    log_action(0, "SEED_EVALUATED", best_loss, best_loss)
    with open(BEST_LOGIC_PATH, "w") as f: f.write(full_module)
    
    dynamic_hints = ""
    meta_hints_path = f"{{MODELS_DIR}}/meta_hints.txt"
    if os.path.exists(meta_hints_path):
        with open(meta_hints_path, "r") as f: dynamic_hints = f.read()

    for iteration in range(1, 101):
        prompt = f"""We are fitting a mathematical model for {domain}.
State vector: {state_vars}.
Time request: {time_req}.
Current ABC-SMC Median MSE: {{best_loss:.6f}}
Previous Run Evolutionary Directions:
{{dynamic_hints}}

Current Dynamics:
```python
{{current_logic}}
```
Hints:
{hints}

Output ONLY a Python code block with `dynamics(t, y, args)` and `metadata`.
CRITICAL SAFETY BOUNDS: Use `jnp.clip(..., -1e6, 1e6)` in `dynamics`.
"""
        reply = call_llm(prompt)
        new_logic, thoughts = extract_code(reply)
        
        # SELF-CORRECTION / REPAIR LOOP
        new_loss = float("inf")
        error_msg = "No code extracted"
        if new_logic:
            new_loss, full_module, error_msg = evaluate_llm_logic(new_logic)
            
            # Agentic Repair
            for attempt in range(1, 3):
                if new_loss < float("inf"):
                    break
                print(f"  [Debugger Agent] Logic failure detected in iteration {{iteration}}. Attempting repair {{attempt}}/2...")
                repair_prompt = f"Your previous code for {domain} failed with this error: `{{error_msg}}`\\n\\nPlease fix the bug and return only the corrected Python code block.\\n\\nPrevious code:\\n```python\\n{{new_logic}}\\n```"
                repair_reply = call_llm(repair_prompt, system_prompt="You are a senior debugging agent. Fix the mathematical code block. Output ONLY the code block.")
                new_logic, _ = extract_code(repair_reply)
                if new_logic:
                    new_loss, full_module, error_msg = evaluate_llm_logic(new_logic)
                else: break

        if not new_logic or new_loss == float("inf"):
            log_action(iteration, f"ERROR_FAILED_REPAIR: {{error_msg[:30]}}", float("inf"), best_loss)
            continue

        with open(f"{{MODELS_DIR}}/proposed_{{iteration}}.py", "w") as f: f.write(full_module)
        if new_loss < best_loss:
            log_action(iteration, "ACCEPTED", new_loss, best_loss)
            best_loss = new_loss
            current_logic = new_logic
            with open(BEST_LOGIC_PATH, "w") as f: f.write(full_module)
        else:
            log_action(iteration, "REJECTED", new_loss, best_loss)

if __name__ == "__main__":
    run_loop()
'''

DATALOADER_TEMPLATE = '''import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

{seed}

def generate_data():
    t0 = 0.0
    t1 = {t1}
    ts = jnp.linspace(t0, t1, {steps})
    y0 = jnp.array({y0})
    args = jnp.array({params})

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/{domain}_ground_truth.npy", gt_data)
    np.save("data/{domain}_time_points.npy", ts)
    print("Saved {domain} ground truth data.")

if __name__ == "__main__":
    generate_data()
'''

def main():
    os.makedirs("automated_science", exist_ok=True)
    for d in DOMAINS:
        # Generate Orchestrator
        orch_code = ORCHESTRATOR_TEMPLATE.format(**d)
        with open(f"automated_science/qwen36_{d['domain']}_orchestrator.py", "w") as f:
            f.write(orch_code)
            
        # Parse t1 and steps from time_req
        import re
        m = re.search(r'(\d+)\s+timepoint', d['time_req'])
        steps = int(m.group(1)) if m else 100
        # rough approx
        t1 = 100.0 if "100" in d['time_req'] else steps/10.0
        
        # Parse params manually
        dl_code = DATALOADER_TEMPLATE.format(
            seed=d['seed'], t1=t1, steps=steps, y0=d['y0'], params=d['params'], domain=d['domain']
        )
        with open(f"automated_science/{d['domain']}_data_loader.py", "w") as f:
            f.write(dl_code)
            
    # Also write a script to run all data loaders
    with open("automated_science/bootstrap_data.sh", "w") as f:
        for d in DOMAINS:
            if not d.get("is_real", False):
                f.write(f"python {d['domain']}_data_loader.py\n")
            
    os.system("chmod +x automated_science/bootstrap_data.sh")
    print("Successfully generated all domains.")

if __name__ == "__main__":
    main()
