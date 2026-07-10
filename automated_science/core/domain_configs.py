from __future__ import annotations

# Configs for all 25 domains
DOMAIN_CONFIGS = {
    "ecology": {
        "name": "Predator-Prey Dynamics (Hudson Bay Lynx-Hare)",
        "y0": [40.0, 9.0],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    r, a, b, m = args
    H, L = y
    dH = r * H - a * H * L
    dL = b * H * L - m * L
    return jnp.array([dH, dL])

metadata = [
    {'name': 'r', 'range': (0.1, 2.0)},
    {'name': 'a', 'range': (0.01, 0.5)},
    {'name': 'b', 'range': (0.01, 0.5)},
    {'name': 'm', 'range': (0.1, 2.0)}
]""",
        "hints": [
            "The real predator-prey dynamics exhibit Allee effects and spatial constraints",
            "Consider Holling Type II or Type III functional responses replacing the linear `a*H*L` term",
            "Carrying capacity on the prey (logistic growth instead of exponential)"
        ],
        "state_desc": "2 elements: [H (Hare), L (Lynx)]",
        "time_desc": "100 yearly timepoints (years 0-100)"
    },
    "oncology": {
        "name": "Tumor Growth Dynamics (Hahnfeldt carrying capacity)",
        "y0": [10.0, 20.0],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    lambda_g, d, e = args
    V, K = y
    # Tumor volume dynamics (Gompertzian-like)
    dV = -lambda_g * V * jnp.log(V / (K + 1e-5) + 1e-5)
    # Carrying capacity dynamics
    dK = d * V - e * K * (V**(2/3))
    return jnp.array([dV, dK])

metadata = [
    {'name': 'lambda_g', 'range': (0.01, 0.5)},
    {'name': 'd', 'range': (0.001, 0.1)},
    {'name': 'e', 'range': (0.001, 0.1)}
]""",
        "hints": [
            "Hahnfeldt models incorporate endothelial cell proliferation and baseline vessel death",
            "Implement Gompertzian growth rather than logistic",
            "K grows via a stimulatory term proportional to V and decays via an inhibitory term"
        ],
        "state_desc": "2 elements: [V (Tumor Volume), K (Carrying Capacity)]",
        "time_desc": "100 timepoints over 50 days"
    },
    "econ": {
        "name": "Macroeconomic Wage-Employment Dynamics (Goodwin)",
        "y0": [0.8, 0.6],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    alpha, beta, gamma, delta = args
    v, u = y
    # Employment rate v, wage share u
    dv = v * (1 - u) - alpha * v
    du = u * (v - beta) - gamma * u
    return jnp.array([dv, du])

metadata = [
    {'name': 'alpha', 'range': (0.01, 0.5)},
    {'name': 'beta', 'range': (0.1, 0.9)},
    {'name': 'gamma', 'range': (0.01, 0.5)},
    {'name': 'delta', 'range': (0.01, 0.5)}
]""",
        "hints": [
            "Integrate a Phillips curve with non-linear inflation expectations",
            "Incorporate time-delays in capital investment",
            "Couple financial debt mechanisms (Minsky financial instability hypothesis)"
        ],
        "state_desc": "2 elements: [v (Employment rate), u (Wage share)]",
        "time_desc": "200 timepoints over 50 years"
    },
    "battery": {
        "name": "Lithium-Ion Battery SEI Layer Growth & Crack Length",
        "y0": [100.0, 1.0],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    k_sei, k_crack = args
    Q, S = y
    # Capacity loss and SEI thickness growth
    dQ = -k_sei * (1.0 / (S + 1e-5)) * Q
    dS = k_sei * (1.0 / (S + 1e-5)) - k_crack * S
    return jnp.array([dQ, dS])

metadata = [
    {'name': 'k_sei', 'range': (0.001, 0.1)},
    {'name': 'k_crack', 'range': (0.001, 0.1)}
]""",
        "hints": [
            "SEI layer growth is proportional to sqrt(time) or cycle number",
            "Implement stress-induced active material loss",
            "Couple volume expansion to SEI cracking mechanisms"
        ],
        "state_desc": "2 elements: [Q (Capacity), S (SEI Layer Thickness)]",
        "time_desc": "500 timepoints representing cycles 0 to 500"
    },
    "pkpd": {
        "name": "Propofol Pharmacokinetics (3-Compartment)",
        "y0": [10.0, 0.0, 0.0],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    k10, k12, k21, k13, k31 = args
    C1, C2, C3 = y
    dC1 = -(k10 + k12 + k13)*C1 + k21*C2 + k31*C3
    dC2 = k12*C1 - k21*C2
    dC3 = k13*C1 - k31*C3
    return jnp.array([dC1, dC2, dC3])

metadata = [
    {'name': 'k10', 'range': (0.01, 0.5)},
    {'name': 'k12', 'range': (0.01, 0.5)},
    {'name': 'k21', 'range': (0.01, 0.5)},
    {'name': 'k13', 'range': (0.01, 0.5)},
    {'name': 'k31', 'range': (0.01, 0.5)}
]""",
        "hints": [
            "Include non-linear drug clearance mechanisms",
            "Look into age-dependent or weight-adjusted covariates modifying inter-compartmental clearance (Q2, Q3)",
            "Add infusion inputs as a forcing function if possible"
        ],
        "state_desc": "3 elements: [C1 (Central), C2 (Fast Periph), C3 (Slow Periph)]",
        "time_desc": "120 timepoints over 120 minutes"
    },
    "synbio": {
        "name": "Synthetic Gene Toggle Switch (Gardner repressor switch)",
        "y0": [2.0, 0.1],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    alpha1, alpha2, beta, gamma = args
    U, V = y
    dU = alpha1 / (1.0 + V**beta) - U
    dV = alpha2 / (1.0 + U**gamma) - V
    return jnp.array([dU, dV])

metadata = [
    {'name': 'alpha1', 'range': (1.0, 10.0)},
    {'name': 'alpha2', 'range': (1.0, 10.0)},
    {'name': 'beta', 'range': (1.0, 5.0)},
    {'name': 'gamma', 'range': (1.0, 5.0)}
]""",
        "hints": [
            "Classic Gardner model uses mutually repressive Hill kinetics",
            "Consider leaky transcription (basal promoter activity)",
            "Consider cooperative binding (higher Hill coefficients) or inducer dynamics"
        ],
        "state_desc": "2 elements: [U (Repressor 1), V (Repressor 2)]",
        "time_desc": "100 timepoints over 50 hours"
    },
    "climate": {
        "name": "Milankovitch Climate Ice Mass Ocean Temp Oscillation",
        "y0": [1.0, 1.0, 1.0],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    a, b, c, d = args
    I, mu, theta = y
    dI = -a * I + b * mu
    dmu = -c * I - d * theta
    dtheta = theta * (1.0 - theta) - I
    return jnp.array([dI, dmu, dtheta])

metadata = [
    {'name': 'a', 'range': (0.1, 1.0)},
    {'name': 'b', 'range': (0.1, 1.0)},
    {'name': 'c', 'range': (0.1, 1.0)},
    {'name': 'd', 'range': (0.1, 1.0)}
]""",
        "hints": [
            "Sea ice albedo feedback introduces extreme non-linearities",
            "Deep ocean carbon sequestration (CO2 lag behind temperature)",
            "Add astronomical Milankovitch forcing as an explicit sinusoidal input"
        ],
        "state_desc": "3 elements: [I (Ice Mass), mu (CO2), theta (Ocean Temp)]",
        "time_desc": "400 timepoints over 400 kyr"
    },
    "cardio": {
        "name": "Cardiovascular Pressure Flow Windkessel Model",
        "y0": [80.0, 0.0],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    R, C = args
    P_ao, Q = y
    dP_ao = (Q - P_ao / R) / C
    dQ = -Q  # Simple decay representing cardiac cycle drop
    return jnp.array([dP_ao, dQ])

metadata = [
    {'name': 'R', 'range': (0.5, 2.0)},
    {'name': 'C', 'range': (0.5, 2.0)}
]""",
        "hints": [
            "2-element Windkessel misses characteristic impedance; try adding it",
            "Introduce non-linear aortic compliance C(P)",
            "Fluid momentum (inertance L) is crucial for accurate dQ/dt"
        ],
        "state_desc": "2 elements: [P_ao (Aortic Pressure), Q (Flow)]",
        "time_desc": "200 timepoints over 2 seconds (high frequency)"
    },
    "bz_chem": {
        "name": "Belousov-Zhabotinsky Chemical Oscillator (Oregonator)",
        "y0": [0.1, 0.1, 0.1],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    q, s, w = args
    X, Y, Z = y
    dX = s * (Y - X * Y + X - q * X**2)
    dY = (Z - Y - X * Y) / s
    dZ = w * (X - Z)
    return jnp.array([dX, dY, dZ])

metadata = [
    {'name': 'q', 'range': (1e-5, 1e-2)},
    {'name': 's', 'range': (1.0, 100.0)},
    {'name': 'w', 'range': (0.1, 10.0)}
]""",
        "hints": [
            "The fundamental reaction exhibits non-linear Oregonator kinetics",
            "Look into autocatalytic steps with quadratic dependencies (X^2 terms)",
            "Consider modifying the stoichiometric factor f"
        ],
        "state_desc": "3 elements: [X (HBrO2), Y (Br-), Z (Ce4+)]",
        "time_desc": "200 timepoints over 50 seconds"
    },
    "epidemiology": {
        "name": "2-Strain Compartmental Epidemiology with Waning Immunity",
        "y0": [0.9, 0.05, 0.05, 0.0],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    beta1, beta2, gamma1, gamma2 = args
    S, I1, I2, R = y
    dS = - (beta1 * I1 + beta2 * I2) * S
    dI1 = beta1 * I1 * S - gamma1 * I1
    dI2 = beta2 * I2 * S - gamma2 * I2
    dR = gamma1 * I1 + gamma2 * I2
    return jnp.array([dS, dI1, dI2, dR])

metadata = [
    {'name': 'beta1', 'range': (0.1, 2.0)},
    {'name': 'beta2', 'range': (0.1, 2.0)},
    {'name': 'gamma1', 'range': (0.05, 0.5)},
    {'name': 'gamma2', 'range': (0.05, 0.5)}
]""",
        "hints": [
            "Waning immunity (R -> S transitions)",
            "Cross-immunity coefficients (infection to I1 limits susceptibility to I2)",
            "Behavioral feedback (transmission rates drop as cases rise)"
        ],
        "state_desc": "4 elements: [S (Susceptible), I1 (Strain 1), I2 (Strain 2), R (Recovered)]",
        "time_desc": "150 timepoints over 150 days"
    },
    "neuro": {
        "name": "Neuronal Action Potential Firing (FitzHugh-Nagumo)",
        "y0": [-1.0, 0.0],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    a, b, tau, I_ext = args
    V, W = y
    dV = V - (V**3)/3.0 - W + I_ext
    dW = (V + a - b * W) / tau
    return jnp.array([dV, dW])

metadata = [
    {'name': 'a', 'range': (0.1, 1.0)},
    {'name': 'b', 'range': (0.1, 1.0)},
    {'name': 'tau', 'range': (0.01, 0.2)},
    {'name': 'I_ext', 'range': (0.1, 1.0)}
]""",
        "hints": [
            "The voltage V exhibits fast cubic nonlinearity.",
            "Recovery W is a slow linear variable."
        ],
        "state_desc": "2 elements: [V (Voltage), W (Recovery)]",
        "time_desc": "200 timepoints over 200 ms"
    },
    "fluid": {
        "name": "Chaotic Fluid Convection (Lorenz)",
        "y0": [1.0, 1.0, 1.0],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    sigma, rho, beta = args
    X, Y, Z = y
    dX = sigma * (Y - X)
    dY = X * (rho - Z) - Y
    dZ = X * Y - beta * Z
    return jnp.array([dX, dY, dZ])

metadata = [
    {'name': 'sigma', 'range': (5.0, 15.0)},
    {'name': 'rho', 'range': (20.0, 35.0)},
    {'name': 'beta', 'range': (1.0, 4.0)}
]""",
        "hints": [
            "The system is highly chaotic and sensitive to initial conditions.",
            "X and Y cross-couple non-linearly."
        ],
        "state_desc": "3 elements: [X (Convection), Y (Horiz Temp), Z (Vert Temp)]",
        "time_desc": "400 timepoints over 40 seconds"
    },
    "astro": {
        "name": "Orbital Gravitational System (Inverse-Square Orbit)",
        "y0": [1.0, 0.0, 0.0, 1.0],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    G_mass = args[0]
    x, y, vx, vy = y
    r = jnp.sqrt(x**2 + y**2 + 1e-5)
    dx = vx
    dy = vy
    dvx = -G_mass * x / (r**3)
    dvy = -G_mass * y / (r**3)
    return jnp.array([dx, dy, dvx, dvy])

metadata = [
    {'name': 'G_mass', 'range': (0.5, 2.0)}
]""",
        "hints": [
            "Gravity follows an inverse-square law based on distance r.",
            "r = sqrt(x^2 + y^2)."
        ],
        "state_desc": "4 elements: [x, y, vx, vy]",
        "time_desc": "500 timepoints over 50 years"
    },
    "chem_kinetics": {
        "name": "Enzyme-Substrate Reversible Catalysis (Michaelis-Menten)",
        "y0": [10.0, 1.0, 0.0, 0.0],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    k_f, k_r, k_cat = args
    S, E, ES, P = y
    dS = -k_f * S * E + k_r * ES
    dE = -k_f * S * E + (k_r + k_cat) * ES
    dES = k_f * S * E - (k_r + k_cat) * ES
    dP = k_cat * ES
    return jnp.array([dS, dE, dES, dP])

metadata = [
    {'name': 'k_f', 'range': (0.01, 1.0)},
    {'name': 'k_r', 'range': (0.01, 1.0)},
    {'name': 'k_cat', 'range': (0.01, 1.0)}
]""",
        "hints": [
            "Total enzyme E + ES is conserved.",
            "Substrate binds reversibly to form ES, which decays to P."
        ],
        "state_desc": "4 elements: [S (Substrate), E (Enzyme), ES (Complex), P (Product)]",
        "time_desc": "100 timepoints over 10 seconds"
    },
    "immune": {
        "name": "Pathogen Replication and T-cell Counteraction",
        "y0": [1.0, 0.1],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    r, a, c, d = args
    P, T = y
    dP = r * P - a * P * T
    dT = c * P * T - d * T
    return jnp.array([dP, dT])

metadata = [
    {'name': 'r', 'range': (0.1, 1.0)},
    {'name': 'a', 'range': (0.01, 0.5)},
    {'name': 'c', 'range': (0.01, 0.5)},
    {'name': 'd', 'range': (0.01, 0.2)}
]""",
        "hints": [
            "Pathogens replicate exponentially until countered by T-cells.",
            "T-cells expand proportionally to pathogen load."
        ],
        "state_desc": "2 elements: [P (Pathogen), T (T-cells)]",
        "time_desc": "150 timepoints over 15 days"
    },
    "pop_genetics": {
        "name": "Population Allele Frequency Selection Model",
        "y0": [0.5],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    s = args[0]
    p = y[0]
    dp = s * p * (1.0 - p)
    return jnp.array([dp])

metadata = [
    {'name': 's', 'range': (0.01, 0.5)}
]""",
        "hints": [
            "Frequency p must stay bounded between 0 and 1.",
            "Selection introduces a p*(1-p) term."
        ],
        "state_desc": "1 element: [p (Allele frequency)]",
        "time_desc": "200 timepoints over 200 generations"
    },
    "thermo": {
        "name": "Stefan-Boltzmann Radiative Cooling Dynamics",
        "y0": [500.0],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    k_cond, sigma_rad = args
    T = y[0]
    T_env = 293.0
    # Cooling = Conduction + Radiation
    dT = -k_cond * (T - T_env) - sigma_rad * (T**4 - T_env**4)
    return jnp.array([dT])

metadata = [
    {'name': 'k_cond', 'range': (0.001, 0.1)},
    {'name': 'sigma_rad', 'range': (1e-12, 1e-9)}
]""",
        "hints": [
            "Linear conduction cooling coupled with Stefan-Boltzmann T^4 radiation cooling."
        ],
        "state_desc": "1 element: [T (Temperature)]",
        "time_desc": "100 timepoints over 100 minutes"
    },
    "materials": {
        "name": "Paris Law Fatigue Crack Propagation Model",
        "y0": [0.01],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    C, m = args
    a = y[0]
    delta_sigma = 100.0  # Applied stress range
    # Stress intensity factor delta_K is proportional to delta_sigma * sqrt(a)
    delta_K = delta_sigma * jnp.sqrt(jnp.pi * a + 1e-6)
    da = C * (delta_K ** m)
    return jnp.array([da])

metadata = [
    {'name': 'C', 'range': (1e-8, 1e-4)},
    {'name': 'm', 'range': (1.5, 4.5)}
]""",
        "hints": [
            "Crack length grows with stress intensity factor raised to power m.",
            "Typically da/dt = C * (dK * sqrt(a))^m."
        ],
        "state_desc": "1 element: [a (Crack Length)]",
        "time_desc": "300 timepoints over 300 cycles"
    },
    "agriculture": {
        "name": "Crop Biomass and Soil Nutrient Coupling",
        "y0": [0.1, 10.0],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    r, K, k_n, d = args
    B, N = y
    dB = r * B * (N / (k_n + N)) * (1.0 - B / K) - d * B
    dN = -r * B * (N / (k_n + N)) + 0.1 * (10.0 - N)
    return jnp.array([dB, dN])

metadata = [
    {'name': 'r', 'range': (0.1, 2.0)},
    {'name': 'K', 'range': (5.0, 50.0)},
    {'name': 'k_n', 'range': (0.5, 5.0)},
    {'name': 'd', 'range': (0.01, 0.5)}
]""",
        "hints": [
            "Biomass grows proportionally to available nutrients.",
            "Nutrients deplete as biomass increases."
        ],
        "state_desc": "2 elements: [B (Biomass), N (Nutrients)]",
        "time_desc": "100 timepoints over 100 days"
    },
    "social": {
        "name": "New Product Adoption Diffusion Model (Bass)",
        "y0": [0.01],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    p, q, M = args
    A = y[0]
    dA = (p + q * (A / M)) * (M - A)
    return jnp.array([dA])

metadata = [
    {'name': 'p', 'range': (0.001, 0.1)},
    {'name': 'q', 'range': (0.01, 0.8)},
    {'name': 'M', 'range': (100.0, 10000.0)}
]""",
        "hints": [
            "Innovation (p) and Imitation (q) drive adoption.",
            "Adoption is capped at Market Size M."
        ],
        "state_desc": "1 element: [A (Adopters)]",
        "time_desc": "50 timepoints over 50 months"
    },
    "real_sunspots": {
        "name": "Solar Sunspot Cycle (Observational Sunspots)",
        "y0": [5.0],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    alpha, beta, gamma = args
    S = y[0]
    dS = alpha * S - beta * S**2 - gamma
    return jnp.array([dS])

metadata = [
    {'name': 'alpha', 'range': (0.01, 1.0)},
    {'name': 'beta', 'range': (0.001, 0.1)},
    {'name': 'gamma', 'range': (0.01, 2.0)}
]""",
        "hints": [
            "Data is highly oscillatory but noisy. Consider an autoregressive-like continuous ODE or a driven oscillator."
        ],
        "state_desc": "1 element: [S (Sunspots)]",
        "time_desc": "309 yearly timepoints"
    },
    "real_nile": {
        "name": "Nile River Volume (Annual Volume)",
        "y0": [1120.0],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    theta, kappa = args
    V = y[0]
    # Simple mean reversion
    dV = -kappa * (V - theta)
    return jnp.array([dV])

metadata = [
    {'name': 'theta', 'range': (800.0, 1300.0)},
    {'name': 'kappa', 'range': (0.01, 1.0)}
]""",
        "hints": [
            "Mean-reverting stochastic-like process. Try Ornstein-Uhlenbeck style deterministic drift."
        ],
        "state_desc": "1 element: [V (Volume)]",
        "time_desc": "100 yearly timepoints"
    },
    "real_macro": {
        "name": "US Real GDP (Quarterly GDP Growth)",
        "y0": [2710.0],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    g, K = args
    G = y[0]
    dG = g * G * (1.0 - G / K)
    return jnp.array([dG])

metadata = [
    {'name': 'g', 'range': (0.001, 0.1)},
    {'name': 'K', 'range': (3000.0, 20000.0)}
]""",
        "hints": [
            "Exponential growth with periodic recessionary shocks."
        ],
        "state_desc": "1 element: [G (GDP)]",
        "time_desc": "203 quarterly timepoints"
    },
    "real_theophylline": {
        "name": "Oral Theophylline Drug Clearance Concentration",
        "y0": [0.74],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    ka, ke = args
    C = y[0]
    # Single compartment kinetics
    dC = ka * jnp.exp(-ka * t) - ke * C
    return jnp.array([dC])

metadata = [
    {'name': 'ka', 'range': (0.1, 5.0)},
    {'name': 'ke', 'range': (0.01, 1.0)}
]""",
        "hints": [
            "Classic 1-compartment oral absorption model. C rises fast, decays exponentially."
        ],
        "state_desc": "1 element: [C (Concentration)]",
        "time_desc": "11 timepoints over 24 hours"
    },
    "real_warfarin_pkpd": {
        "name": "Warfarin PK/PD Concentration And Response",
        "y0": [0.0, 100.0],
        "dt0": 0.25,
        "max_steps": 10000,
        "target_samples": 200,
        "generations": 8,
        "initial_particles": 50000,
        "evaluation_mode": "dense_masked",
        "real_data_adapter": "warfarin_pkpd",
        "normalized_data_path": "data/real/pkpd/warfarin_normalized.csv",
        "subject_holdout_path": "data/real/pkpd/subject_holdout.json",
        "observation_mode": "masked_endpoints",
        "seed_logic": """def dynamics(t, y, args):
    ka, ke, V, E0, kout, imax, ic50 = args
    C, E = y
    absorption = (100.0 * ka / (V + 1e-6)) * jnp.exp(-ka * t)
    inhibition = imax * C / (ic50 + C + 1e-6)
    dC = absorption - ke * C
    dE = kout * (E0 * (1.0 - inhibition) - E)
    return jnp.clip(jnp.array([dC, dE]), -1e6, 1e6)

metadata = [
    {'name': 'ka', 'range': (0.01, 5.0)},
    {'name': 'ke', 'range': (0.001, 1.0)},
    {'name': 'V', 'range': (1.0, 200.0)},
    {'name': 'E0', 'range': (20.0, 150.0)},
    {'name': 'kout', 'range': (0.001, 1.0)},
    {'name': 'imax', 'range': (0.01, 1.0)},
    {'name': 'ic50', 'range': (0.001, 20.0)}
]""",
        "hints": [
            "This seed uses a dense concentration/response bridge for the current sandbox; endpoint-masked sparse evaluation is the next required upgrade.",
            "Consider effect-compartment lag or indirect response turnover variants.",
            "Weight-scaled volume or clearance may explain subject-level held-out failures once subject holdout is available."
        ],
        "state_desc": "2 elements: [C (Warfarin concentration), E (Prothrombin/PCA response)]",
        "time_desc": "Sparse warfarin PK/PD observations in hours; fixture uses 0-72 hours"
    },
    "real_battery_nasa_capacity": {
        "name": "NASA Li-Ion Battery Capacity Fade",
        "y0": [1.0],
        "dt0": 1.0,
        "max_steps": 10000,
        "target_samples": 200,
        "generations": 8,
        "initial_particles": 50000,
        "real_data_adapter": "battery_nasa",
        "normalized_data_path": "data/real/battery_nasa/cycle_level.csv",
        "evaluation_mode": "dense",
        "seed_logic": """def dynamics(t, y, args):
    k_sqrt, tau, k_lin = args
    Q = y[0]
    cycle = jnp.maximum(t, 0.0)
    fade = k_sqrt / jnp.sqrt(cycle + tau) + k_lin * Q
    dQ = -fade
    return jnp.array([dQ])

metadata = [
    {'name': 'k_sqrt', 'range': (1e-5, 5e-2)},
    {'name': 'tau', 'range': (0.1, 100.0)},
    {'name': 'k_lin', 'range': (1e-6, 1e-2)}
]""",
        "hints": [
            "Fixture uses a one-cell SOH capacity bridge; real promotion should use downloaded NASA PCoE battery files.",
            "Consider square-root SEI fade, linear active material loss, knee acceleration, and temperature-conditioned rates.",
            "Only add resistance-coupled dynamics after Re/Rct joins are validated."
        ],
        "state_desc": "1 element: [Q (state of health / normalized capacity)]",
        "time_desc": "Discharge cycle index from NASA PCoE Li-ion battery aging data"
    },
    "real_co2": {
        "name": "Atmospheric CO2 Concentration Growth",
        "y0": [315.0],
        "dt0": 0.5,
        "max_steps": 10000,
        "seed_logic": """def dynamics(t, y, args):
    r, K = args
    C = y[0]
    dC = r * C * (1.0 - C / K)
    return jnp.array([dC])

metadata = [
    {'name': 'r', 'range': (0.001, 0.1)},
    {'name': 'K', 'range': (350.0, 600.0)}
]""",
        "hints": [
            "Accelerating exponential growth."
        ],
        "state_desc": "1 element: [C (CO2)]",
        "time_desc": "43 yearly timepoints"
    }
}
