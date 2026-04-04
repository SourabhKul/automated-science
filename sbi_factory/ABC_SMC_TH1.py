# A python program to run the ABC-SMC for the Th-1 cell type, hypothesis 1 using the results from the RPE1 cell type.
# "Approximate Bayesian computation scheme for parameter inference and model selection in dynamical systems" - reference.

# Look the necessary modules.
import numpy as np
import time
from scipy import integrate

# Read in the experimental data.
IL6_data_pS1 = np.loadtxt("IL6_data_pS1_mean_TH1.txt")
IL6_data_pS3 = np.loadtxt("IL6_data_pS3_mean_TH1.txt")
IL27_data_pS1 = np.loadtxt("IL27_data_pS1_mean_TH1.txt")
IL27_data_pS3 = np.loadtxt("IL27_data_pS3_mean_TH1.txt")
IL6_final = [IL6_data_pS1, IL6_data_pS3]
IL27_final = [IL27_data_pS1, IL27_data_pS3]

RPE_posteriors = np.loadtxt("RPE1_posteriors.txt")


# Define the ordinary differential equation models.
# Hyper IL6 mathematical model where the terms involving the parameter beta apply only to hypothesis 1 and
# the terms involving the parameter gamma apply only to hypothesis 2.
def dIL6_dt(R, t, par):
    (
        k1ap,
        k1am,
        k3ap,
        k3am,
        qx,
        d1,
        d3,
        r10,
        s10,
        s30,
        r1p,
        r1m,
        r2p,
        r2m,
        beta,
        gamma,
    ) = par.T
    return np.array(
        [
            -r1p * R[0] * R[1]
            + r1m * R[2]
            - beta * R[0]
            - gamma * (R[20] + R[21]) * R[0],
            -r1p * R[0] * R[1] + r1m * R[2],
            r1p * R[0] * R[1]
            - r1m * R[2]
            - 2 * r2p * pow(R[2], 2)
            + 2 * r2m * R[3]
            - beta * R[2]
            - gamma * (R[20] + R[21]) * R[2],
            r2p * pow(R[2], 2)
            - r2m * R[3]
            - 2 * k1ap * R[3] * R[4]
            + k1am * R[6]
            - 2 * k3ap * R[3] * R[5]
            + k3am * R[7]
            + k1am * R[8]
            + k3am * R[9]
            - beta * R[3]
            - gamma * (R[20] + R[21]) * R[3],
            -k1ap * R[4] * (R[6] + R[7] + R[8] + R[9])
            + k1am * (2 * R[10] + R[16] + R[11] + R[18])
            - 2 * k1ap * R[3] * R[4]
            + k1am * R[6]
            + d1 * R[20],
            -k3ap * R[5] * (R[6] + R[7] + R[8] + R[9])
            + k3am * (2 * R[13] + R[16] + R[14] + R[17])
            - 2 * k3ap * R[3] * R[5]
            + k3am * R[7]
            + d3 * R[21],
            2 * k1ap * R[3] * R[4]
            - k1am * R[6]
            - k1ap * R[6] * R[4]
            + 2 * k1am * R[10]
            - k3ap * R[6] * R[5]
            + k3am * R[16]
            - qx * R[6]
            + k1am * R[11]
            + k3am * R[18]
            - beta * R[6]
            - gamma * (R[20] + R[21]) * R[6],
            2 * k3ap * R[3] * R[5]
            - k3am * R[7]
            - k3ap * R[7] * R[5]
            + 2 * k3am * R[13]
            - k1ap * R[7] * R[4]
            + k1am * R[16]
            - qx * R[7]
            + k3am * R[14]
            + k1am * R[17]
            - beta * R[7]
            - gamma * (R[20] + R[21]) * R[7],
            -k1ap * R[4] * R[8]
            + k1am * R[11]
            - k3ap * R[5] * R[8]
            + k3am * R[17]
            + qx * R[6]
            - k1am * R[8]
            + 2 * k1am * R[12]
            + k3am * R[19]
            - beta * R[8]
            - gamma * (R[20] + R[21]) * R[8],
            -k3ap * R[5] * R[9]
            + k3am * R[14]
            - k1ap * R[4] * R[9]
            + k1am * R[18]
            + qx * R[7]
            - k3am * R[9]
            + 2 * k3am * R[15]
            + k1am * R[19]
            - beta * R[9]
            - gamma * (R[20] + R[21]) * R[9],
            k1ap * R[4] * R[6]
            - 2 * k1am * R[10]
            - 2 * qx * R[10]
            - beta * R[10]
            - gamma * (R[20] + R[21]) * R[10],
            k1ap * R[8] * R[4]
            - k1am * R[11]
            + 2 * qx * R[10]
            - (qx + k1am) * R[11]
            - beta * R[11]
            - gamma * (R[20] + R[21]) * R[11],
            qx * R[11]
            - 2 * k1am * R[12]
            - beta * R[12]
            - gamma * (R[20] + R[21]) * R[12],
            k3ap * R[5] * R[7]
            - 2 * k3am * R[13]
            - 2 * qx * R[13]
            - beta * R[13]
            - gamma * (R[20] + R[21]) * R[13],
            k3ap * R[9] * R[5]
            - k3am * R[14]
            + 2 * qx * R[13]
            - (qx + k3am) * R[14]
            - beta * R[14]
            - gamma * (R[20] + R[21]) * R[14],
            qx * R[14]
            - 2 * k3am * R[15]
            - beta * R[15]
            - gamma * (R[20] + R[21]) * R[15],
            k1ap * R[4] * R[7]
            - k1am * R[16]
            + k3ap * R[6] * R[5]
            - k3am * R[16]
            - (qx + qx) * R[16]
            - beta * R[16]
            - gamma * (R[20] + R[21]) * R[16],
            qx * R[16]
            + k3ap * R[8] * R[5]
            - k3am * R[17]
            - qx * R[17]
            - k1am * R[17]
            - beta * R[17]
            - gamma * (R[20] + R[21]) * R[17],
            qx * R[16]
            + k1ap * R[9] * R[4]
            - k1am * R[18]
            - qx * R[18]
            - k3am * R[18]
            - beta * R[18]
            - gamma * (R[20] + R[21]) * R[18],
            qx * R[18]
            + qx * R[17]
            - k1am * R[19]
            - k3am * R[19]
            - beta * R[19]
            - gamma * (R[20] + R[21]) * R[19],
            k1am * (R[8] + R[11] + R[17] + R[19]) + 2 * k1am * R[12] - d1 * R[20],
            k3am * (R[9] + R[14] + R[18] + R[19]) + 2 * k3am * R[15] - d3 * R[21],
        ]
    )


# IL27 mathematical model where the terms involving the parameter beta apply only to hypothesis 1 and
# the terms involving the parameter gamma apply only to hypothesis 2.
def dIL27_dt(R, t, par):
    (
        k1ap,
        k1am,
        k1bp,
        k1bm,
        k3ap,
        k3am,
        k3bp,
        k3bm,
        qx,
        d1,
        d3,
        r10,
        r20,
        s10,
        s30,
        r1p,
        r1m,
        r2p,
        r2m,
        beta,
        gamma,
    ) = par.T
    return np.array(
        [
            -r2p * R[0] * R[3]
            + r2m * R[4]
            - beta * R[0]
            - gamma * (R[31] + R[32]) * R[0],
            -r1p * R[1] * R[2]
            + r1m * R[3]
            - beta * R[1]
            - gamma * (R[31] + R[32]) * R[1],
            -r1p * R[1] * R[2] + r1m * R[3],
            r1p * R[1] * R[2]
            - r1m * R[3]
            - r2p * R[0] * R[3]
            + r2m * R[4]
            - beta * R[3]
            - gamma * (R[31] + R[32]) * R[3],
            r2p * R[0] * R[3]
            - r2m * R[4]
            - (k1ap + k1bp) * R[4] * R[5]
            + k1am * R[7]
            + k1bm * R[8]
            - (k3ap + k3bp) * R[4] * R[6]
            + k3am * R[9]
            + k3bm * R[10]
            + k1am * R[11]
            + k1bm * R[12]
            + k3am * R[13]
            + k3bm * R[14]
            - beta * R[4]
            - gamma * (R[31] + R[32]) * R[4],
            -k1ap * R[5] * (R[4] + R[8] + R[12] + R[10] + R[14])
            + k1am * (R[7] + R[15] + R[17] + R[23] + R[27])
            - k1bp * R[5] * (R[4] + R[7] + R[11] + R[9] + R[13])
            + k1bm * (R[8] + R[15] + R[16] + R[24] + R[26])
            + d1 * R[31],
            -k3ap * R[6] * (R[4] + R[8] + R[12] + R[10] + R[14])
            + k3am * (R[9] + R[24] + R[28] + R[19] + R[21])
            - k3bp * R[6] * (R[4] + R[7] + R[11] + R[9] + R[13])
            + k3bm * (R[10] + R[23] + R[25] + R[19] + R[20])
            + d3 * R[32],
            k1ap * R[5] * R[4]
            - k1am * R[7]
            - k1bp * R[5] * R[7]
            + k1bm * R[15]
            - k3bp * R[6] * R[7]
            + k3bm * R[23]
            - qx * R[7]
            + k1bm * R[17]
            + k3bm * R[27]
            - beta * R[7]
            - gamma * (R[31] + R[32]) * R[7],
            k1bp * R[5] * R[4]
            - k1bm * R[8]
            - k1ap * R[5] * R[8]
            + k1am * R[15]
            - k3ap * R[6] * R[8]
            + k3am * R[24]
            - qx * R[8]
            + k1am * R[16]
            + k3am * R[26]
            - beta * R[8]
            - gamma * (R[31] + R[32]) * R[8],
            k3ap * R[6] * R[4]
            - k3am * R[9]
            - k3bp * R[6] * R[9]
            + k3bm * R[19]
            - k1bp * R[5] * R[9]
            + k1bm * R[24]
            - qx * R[9]
            + k3bm * R[21]
            + k1bm * R[28]
            - beta * R[9]
            - gamma * (R[31] + R[32]) * R[9],
            k3bp * R[6] * R[4]
            - k3bm * R[10]
            - k3ap * R[6] * R[10]
            + k3am * R[19]
            - k1ap * R[5] * R[10]
            + k1am * R[23]
            - qx * R[10]
            + k3am * R[20]
            + k1am * R[25]
            - beta * R[10]
            - gamma * (R[31] + R[32]) * R[10],
            -k1bp * R[11] * R[5]
            + k1bm * R[16]
            - k3bp * R[11] * R[6]
            + k3bm * R[25]
            + qx * R[7]
            - k1am * R[11]
            + k1bm * R[18]
            + k3bm * R[29]
            - beta * R[11]
            - gamma * (R[31] + R[32]) * R[11],
            -k1ap * R[12] * R[5]
            + k1am * R[17]
            - k3ap * R[12] * R[6]
            + k3am * R[28]
            + qx * R[8]
            - k1bm * R[12]
            + k1am * R[18]
            + k3am * R[30]
            - beta * R[12]
            - gamma * (R[31] + R[32]) * R[12],
            -k3bp * R[13] * R[6]
            + k3bm * R[20]
            - k1bp * R[13] * R[5]
            + k1bm * R[26]
            + qx * R[9]
            - k3am * R[13]
            + k3bm * R[22]
            + k1bm * R[30]
            - beta * R[13]
            - gamma * (R[31] + R[32]) * R[13],
            -k3ap * R[14] * R[6]
            + k3am * R[21]
            - k1ap * R[14] * R[5]
            + k1am * R[27]
            + qx * R[10]
            - k3bm * R[14]
            + k3am * R[22]
            + k1am * R[29]
            - beta * R[14]
            - gamma * (R[31] + R[32]) * R[14],
            k1ap * R[5] * R[8]
            - k1am * R[15]
            + k1bp * R[7] * R[5]
            - k1bm * R[15]
            - (qx + qx) * R[15]
            - beta * R[15]
            - gamma * (R[31] + R[32]) * R[15],
            k1bp * R[11] * R[5]
            - k1bm * R[16]
            + qx * R[15]
            - qx * R[16]
            - k1am * R[16]
            - beta * R[16]
            - gamma * (R[31] + R[32]) * R[16],
            k1ap * R[5] * R[12]
            - k1am * R[17]
            + qx * R[15]
            - qx * R[17]
            - k1bm * R[17]
            - beta * R[17]
            - gamma * (R[31] + R[32]) * R[17],
            qx * R[17]
            + qx * R[16]
            - (k1am + k1bm) * R[18]
            - beta * R[18]
            - gamma * (R[31] + R[32]) * R[18],
            k3ap * R[6] * R[10]
            - k3am * R[19]
            + k3bp * R[9] * R[6]
            - k3bm * R[19]
            - (qx + qx) * R[19]
            - beta * R[19]
            - gamma * (R[31] + R[32]) * R[19],
            k3bp * R[13] * R[6]
            - k3bm * R[20]
            + qx * R[19]
            - qx * R[20]
            - k3am * R[20]
            - beta * R[20]
            - gamma * (R[31] + R[32]) * R[20],
            k3ap * R[6] * R[14]
            - k3am * R[21]
            + qx * R[19]
            - qx * R[21]
            - k3bm * R[21]
            - beta * R[21]
            - gamma * (R[31] + R[32]) * R[21],
            qx * R[21]
            + qx * R[20]
            - (k3am + k3bm) * R[22]
            - beta * R[22]
            - gamma * (R[31] + R[32]) * R[22],
            k1ap * R[5] * R[10]
            - k1am * R[23]
            + k3bp * R[7] * R[6]
            - k3bm * R[23]
            - (qx + qx) * R[23]
            - beta * R[23]
            - gamma * (R[31] + R[32]) * R[23],
            k3ap * R[6] * R[8]
            - k3am * R[24]
            + k1bp * R[9] * R[5]
            - k1bm * R[24]
            - (qx + qx) * R[24]
            - beta * R[24]
            - gamma * (R[31] + R[32]) * R[24],
            k3bp * R[11] * R[6]
            - k3bm * R[25]
            + qx * R[23]
            - qx * R[25]
            - k1am * R[25]
            - beta * R[25]
            - gamma * (R[31] + R[32]) * R[25],
            k1bp * R[13] * R[5]
            - k1bm * R[26]
            + qx * R[24]
            - qx * R[26]
            - k3am * R[26]
            - beta * R[26]
            - gamma * (R[31] + R[32]) * R[26],
            k1ap * R[5] * R[14]
            - k1am * R[27]
            + qx * R[23]
            - qx * R[27]
            - k3bm * R[27]
            - beta * R[27]
            - gamma * (R[31] + R[32]) * R[27],
            k3ap * R[6] * R[12]
            - k3am * R[28]
            + qx * R[24]
            - qx * R[28]
            - k1bm * R[28]
            - beta * R[28]
            - gamma * (R[31] + R[32]) * R[28],
            qx * R[27]
            + qx * R[25]
            - (k1am + k3bm) * R[29]
            - beta * R[29]
            - gamma * (R[31] + R[32]) * R[29],
            qx * R[28]
            + qx * R[26]
            - (k3am + k1bm) * R[30]
            - beta * R[30]
            - gamma * (R[31] + R[32]) * R[30],
            k1am * (R[11] + R[16] + R[18] + R[25] + R[29])
            + k1bm * (R[12] + R[17] + R[18] + R[28] + R[30])
            - d1 * R[31],
            k3am * (R[13] + R[26] + R[30] + R[20] + R[22])
            + k3bm * (R[14] + R[27] + R[29] + R[21] + R[22])
            - d3 * R[32],
        ]
    )


# Set the timecourse for the integration (units of seconds).
tt = np.linspace(0.0, 10800, 18001)


# Function for the first iteration of the ABC-SMC where the parameters are sampled from the prior distributions.
def first_iteration(N, ep):
    print("Running iteration: 0")
    epsilon = ep
    # Empty arrays for the accepted parameters.
    accepted_params_hyp1 = np.empty((0, 25))
    # Store the distance measures from the ABC.
    results_hyp1 = np.empty((0))
    number = 0  # Counter for the population.
    truns = 0  # Count total number of runs in order to acceptance percentage.
    while number < N:  # Run the loop until N parameter sets are accepted.
        truns += 1
        # Draw the parameters either from the prior distributions or from the RPE1 posteriors (for those parameters
        # which do not vary due to cell type).
        setx = np.random.randint(
            0, len(RPE_posteriors)
        )  # Randomly select an RPE1 parameter set.
        k1ap = RPE_posteriors[
            setx, 10
        ]  # np.power(10, np.random.uniform(-7, 1, batch_size))
        k1am = RPE_posteriors[setx, 11]
        k1bp = RPE_posteriors[setx, 12]
        k1bm = RPE_posteriors[setx, 13]
        k3ap = RPE_posteriors[setx, 14]
        k3am = RPE_posteriors[setx, 15]
        k3bp = RPE_posteriors[setx, 16]
        k3bm = RPE_posteriors[setx, 17]
        qx = pow(10, np.random.uniform(-3, 2))
        d1 = pow(10, np.random.uniform(-5, -2))
        d3 = pow(10, np.random.uniform(-5, -2))
        r10 = np.random.normal(12.7, 6.35)
        r20 = np.random.normal(33.8, 16.9)
        s10 = np.random.normal(300, 100)
        s30 = np.random.normal(400, 100)
        outputs = []  # Empty list for the model outputs.
        paramsa = []  # Empty list for the parameters.
        for stim in range(2):  # Loop for the two cytokine models.
            # Run the HypIL-6 model.
            if stim == 0:
                r1p = RPE_posteriors[setx, 0]
                r1m = RPE_posteriors[setx, 1]
                r2p = RPE_posteriors[setx, 2]
                r2m = RPE_posteriors[setx, 3]
                beta = pow(10, np.random.uniform(-5, -1, batch_size))
                paramsa.append(np.array((r1p, r1m, r2p, r2m, beta)))

                parset = np.array(
                    (
                        k1ap,
                        k1am,
                        k3ap,
                        k3am,
                        qx,
                        d1,
                        d3,
                        r10,
                        s10,
                        s30,
                        r1p,
                        r1m,
                        r2p,
                        r2m,
                        beta,
                        0,
                    )
                )  # Parameter set to feed into the integrator.
                R0 = np.array(
                    [
                        r10,
                        10,
                        0,
                        0,
                        s10,
                        s30,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                )  # Initial concentrations.
                R = integrate.odeint(
                    dIL6_dt, R0, tt, args=(parset,)
                )  # Integrate the mathematical model.
                (
                    n1,
                    n2,
                    n3,
                    n4,
                    n5,
                    n6,
                    n7,
                    n8,
                    n9,
                    n10,
                    n11,
                    n12,
                    n13,
                    n14,
                    n15,
                    n16,
                    n17,
                    n18,
                    n19,
                    n20,
                    n21,
                    n22,
                ) = R.T  # Model outputs for each variable.
                pS1 = (
                    n9 + n12 + 2 * n13 + n18 + n20 + n21
                )  # Sum of variables containing pSTAT1.
                pS3 = (
                    n10 + n15 + 2 * n16 + n19 + n20 + n22
                )  # Sum of vairables containing pSTAT3.
            # Run the IL-27 model.
            else:
                r1p = RPE_posteriors[setx, 5]
                r1m = RPE_posteriors[setx, 6]
                r2p = RPE_posteriors[setx, 7]
                r2m = RPE_posteriors[setx, 8]
                beta = pow(10, np.random.uniform(-5, -1))
                paramsa.append(np.array((r1p, r1m, r2p, r2m, beta)))

                parset = np.array(
                    (
                        k1ap,
                        k1am,
                        k1bp,
                        k1bm,
                        k3ap,
                        k3am,
                        k3bp,
                        k3bm,
                        qx,
                        d1,
                        d3,
                        r10,
                        r20,
                        s10,
                        s30,
                        r1p,
                        r1m,
                        r2p,
                        r2m,
                        beta,
                        0,
                    )
                )  # Parameter set to feed into the integrator.
                R0 = np.array(
                    [
                        r10,
                        r20,
                        2,
                        0,
                        0,
                        s10,
                        s30,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                )  # Initial concentrations.
                R = integrate.odeint(
                    dIL27_dt, R0, tt, args=(parset,)
                )  # Integrate the mathematical model.
                (
                    n1,
                    n2,
                    n3,
                    n4,
                    n5,
                    n6,
                    n7,
                    n8,
                    n9,
                    n10,
                    n11,
                    n12,
                    n13,
                    n14,
                    n15,
                    n16,
                    n17,
                    n18,
                    n19,
                    n20,
                    n21,
                    n22,
                    n23,
                    n24,
                    n25,
                    n26,
                    n27,
                    n28,
                    n29,
                    n30,
                    n31,
                    n32,
                    n33,
                ) = R.T  # Model outputs for each vairable.
                pS1 = (
                    n12 + n13 + n17 + n18 + 2 * n19 + n26 + n29 + n30 + n31 + n32
                )  # Sum of variables containing pSTAT1.
                pS3 = (
                    n14 + n15 + n21 + n22 + 2 * n23 + n27 + n28 + n30 + n31 + n33
                )  # Sum of variables containing pSTAT3.
            # Add model simulations for total pSTAT1 and pSTAT3 to outputs list.
            outputs.append(
                np.array((pS1[0::100], pS3[0::100]))
            )  # Append model outputs (cut to length 181) to the list.

        paramsa.append(
            np.array(
                (
                    k1ap,
                    k1am,
                    k1bp,
                    k1bm,
                    k3ap,
                    k3am,
                    k3bp,
                    k3bm,
                    qx,
                    d1,
                    d3,
                    r10,
                    r20,
                    s10,
                    s30,
                )
            )
        )
        paramsa = np.hstack(
            (paramsa)
        )  # Create an array of the sampled parameter values.

        # Take the values of the model simulations that correlate to the data time points.
        time_points = np.array((0, 5, 15, 30, 60, 90, 120, 180))  # Data time points.
        IL6_pS1_model = np.empty((8,))
        IL6_pS3_model = np.empty((8,))
        IL27_pS1_model = np.empty((8,))
        IL27_pS3_model = np.empty((8,))
        for i in range(8):
            IL6_pS1_model[i] = outputs[0][0, :][time_points[i]]
            IL6_pS3_model[i] = outputs[0][1, :][time_points[i]]
            IL27_pS1_model[i] = outputs[1][0, :][time_points[i]]
            IL27_pS3_model[i] = outputs[1][1, :][time_points[i]]

        # Normalise the model in the same way as the data by dividing by IL-27 time 30 minutes.
        norm_point_s1 = IL27_pS1_model[3]
        norm_point_s3 = IL27_pS3_model[3]
        IL6_pS1_model_norm = IL6_pS1_model / norm_point_s1
        IL6_pS3_model_norm = IL6_pS3_model / norm_point_s3
        IL27_pS1_model_norm = IL27_pS1_model / norm_point_s1
        IL27_pS3_model_norm = IL27_pS3_model / norm_point_s3

        # Compute the distance between the model and the data and append it to the results list.
        IL6_pS1_dis = []
        IL6_pS3_dis = []
        IL27_pS1_dis = []
        IL27_pS3_dis = []
        for t in range(8):
            IL6_pS1_dis.append(pow((IL6_pS1_model_norm[t] - IL6_final[0][t]), 2))
            IL6_pS3_dis.append(pow((IL6_pS3_model_norm[t] - IL6_final[1][t]), 2))
            IL27_pS1_dis.append(pow((IL27_pS1_model_norm[t] - IL27_final[0][t]), 2))
            IL27_pS3_dis.append(pow((IL27_pS3_model_norm[t] - IL27_final[1][t]), 2))
        dist = np.sum(
            (
                np.array(
                    (
                        np.sum(IL6_pS1_dis),
                        np.sum(IL6_pS3_dis),
                        np.sum(IL27_pS1_dis),
                        np.sum(IL27_pS3_dis),
                    )
                )
            )
        )
        dist2 = np.sqrt(dist)

        # If the distance if less than the first value of epsilon, append the parameters to the accepted parameters array,
        # and add one to the parameter set counter.
        if dist2 < epsilon:
            number += 1
            results_hyp1 = np.hstack((results_hyp1, dist2))
            accepted_params_hyp1 = np.vstack((accepted_params_hyp1, paramsa))

    # Compute the weight for each accepted parameter set (at iteration 1, the parameter sets are equally weighted).
    weights_hyp1 = np.empty((0, 1))
    for i in range(len(accepted_params_hyp1)):
        weights_hyp1 = np.vstack((weights_hyp1, 1 / len(accepted_params_hyp1)))

    # Print information about the first run of the ABC.
    print("Acceptance rate for iteration 0: " + str(N * 100 / truns))
    print("Epsilon = " + str(epsilon))
    print("Total runs = " + str(truns))
    # Return the results (distances, accepted parameters, weights, total number of runs).
    return [
        np.hstack(
            (
                np.reshape(results_hyp1, (len(accepted_params_hyp1), 1)),
                accepted_params_hyp1,
                weights_hyp1,
            )
        ),
        truns,
    ]


# Function for the other iterations of the ABC-SMC where the parameters are sampled from the previous posteriors.
def other_iterations(N, it):
    print("Running iteration: " + str(it + 1))
    # Get the value of epsilon from the list.
    epsilon = epsilons[it]
    p_list = [i for i in range(N)]

    # Upper and lower bounds for the uniform distributions of the priors for each parameter.
    lower_bounds = [
        -10,
        -15,
        -2,
        -3,
        -5,
        -10,
        -15,
        -2,
        -3,
        -5,
        -7,
        -2,
        -7,
        -2,
        -7,
        -2,
        -7,
        -2,
        -3,
        -5,
        -5,
        0,
        0,
        0,
        0,
    ]
    upper_bounds = [
        5,
        5,
        3,
        1,
        -1,
        5,
        5,
        3,
        1,
        -1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        -2,
        -2,
        40,
        100,
        800,
        1000,
    ]

    # Compute uniform areas to sample within in order to perturb the parameters.
    ranges = []
    for i in range(25):
        if i in [4, 9, 18, 19, 20, 21, 22, 23, 24]:
            if i in [21, 22, 23, 24]:
                r1 = np.max(ABC_runs[it][:, i + 1]) - np.min(ABC_runs[it][:, i + 1])
            else:
                r1 = np.max(np.log10(ABC_runs[it][:, i + 1])) - np.min(
                    np.log10(ABC_runs[it][:, i + 1])
                )
            ranges.append(r1)
        else:
            r1 = upper_bounds[i] - lower_bounds[i]
            ranges.append(r1)

    ranges_arr = np.asarray(ranges)
    sigma = []
    for i in range(25):
        if i in [4, 9, 18, 19, 20, 21, 22, 23, 24]:
            sigma.append((0.2 * ranges_arr[i]))
        else:
            sigma.append((ranges_arr[i]))
    sigma = np.asarray((sigma))

    # Empty arrays for the prior samples.
    priors_hyp1 = np.empty((0, 25))
    # Empty arrays for the accepted parameters.
    accepted_params_hyp1 = np.empty((0, 25))
    # Store the distance measures from the ABC.
    results_hyp1 = np.empty((0))
    # Store the weights.
    weights_hyp1 = np.empty((0))

    number = 0  # Counter for the population.
    truns = 0  # Count total number of runs in order to acceptance percentage.
    while number < N:  # Run the loop until N parameter sets are accepted.
        truns += 1
        # Loop to sample parameters from the posterior distributions of the previous iteration. The parameters are then perturbed,
        # using a uniform perturbation kernel, and if the new parameters all lie within their uniform priors, then they are used in the model integration,
        # otherwise they are resampled.
        check = 0
        while check < 1:
            choice = np.random.choice(
                p_list, 1, p=ABC_runs[it][:, 26]
            )  # Choose a random parameter set from previous iterations posteriors.
            prior_sample = ABC_runs[it][:, range(1, 26)][choice]
            # Generate new parameters through perturbation.
            parameters = []
            for i in range(25):
                if i in [4, 9, 18, 19, 20, 21, 22, 23, 24]:
                    if i in [21, 22, 23, 24]:
                        lower = prior_sample[0, i] - sigma[i]
                        upper = prior_sample[0, i] + sigma[i]
                    else:
                        lower = np.log10(prior_sample[0, i]) - sigma[i]
                        upper = np.log10(prior_sample[0, i]) + sigma[i]
                    parameter = np.random.uniform(lower, upper)
                    if i in [21, 22, 23, 24]:
                        parameters.append(parameter)
                    else:
                        parameters.append(pow(10, parameter))
                else:
                    parameter = np.random.choice(RPE_posteriors[:, i])
                    parameters.append(parameter)
            # Check that the new parameters are feasible given the priors.
            check_out = 0
            for ik in range(25):
                if ik in [21, 22, 23, 24]:
                    if (
                        parameters[ik] < lower_bounds[ik]
                        or parameters[ik] > upper_bounds[ik]
                    ):
                        check_out = 1
                else:
                    if parameters[ik] < pow(10, lower_bounds[ik]) or parameters[
                        ik
                    ] > pow(10, upper_bounds[ik]):
                        check_out = 1
            if check_out == 0:
                check += 1

        k1ap = parameters[10]
        k1am = parameters[11]
        k1bp = parameters[12]
        k1bm = parameters[13]
        k3ap = parameters[14]
        k3am = parameters[15]
        k3bp = parameters[16]
        k3bm = parameters[17]
        qx = parameters[18]
        d1 = parameters[19]
        d3 = parameters[20]
        r10 = parameters[21]
        r20 = parameters[22]
        s10 = parameters[23]
        s30 = parameters[24]
        outputs = []  # Empty list for the model outputs.
        paramsa = []  # Empty list for the parameters.
        for stim in range(2):  # Loop for the two cytokine models.
            # Run the HypIL-6 model.
            if stim == 0:
                r1p = parameters[0]
                r1m = parameters[1]
                r2p = parameters[2]
                r2m = parameters[3]
                beta = parameters[4]
                paramsa.append(np.array((r1p, r1m, r2p, r2m, beta)))

                parset = np.array(
                    (
                        k1ap,
                        k1am,
                        k3ap,
                        k3am,
                        qx,
                        d1,
                        d3,
                        r10,
                        s10,
                        s30,
                        r1p,
                        r1m,
                        r2p,
                        r2m,
                        beta,
                        0,
                    )
                )  # Parameter set to feed into the integrator.
                R0 = np.array(
                    [
                        r10,
                        10,
                        0,
                        0,
                        s10,
                        s30,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                )  # Initial concentrations.
                R = integrate.odeint(
                    dIL6_dt, R0, tt, args=(parset,)
                )  # Integrate the mathematical model.
                (
                    n1,
                    n2,
                    n3,
                    n4,
                    n5,
                    n6,
                    n7,
                    n8,
                    n9,
                    n10,
                    n11,
                    n12,
                    n13,
                    n14,
                    n15,
                    n16,
                    n17,
                    n18,
                    n19,
                    n20,
                    n21,
                    n22,
                ) = R.T  # Model outputs for each variable.
                pS1 = (
                    n9 + n12 + 2 * n13 + n18 + n20 + n21
                )  # Sum of variables containing pSTAT1.
                pS3 = (
                    n10 + n15 + 2 * n16 + n19 + n20 + n22
                )  # Sum of vairables containing pSTAT3.
            # Run the IL-27 model.
            else:
                r1p = parameters[5]
                r1m = parameters[6]
                r2p = parameters[7]
                r2m = parameters[8]
                beta = parameters[9]
                paramsa.append(np.array((r1p, r1m, r2p, r2m, beta)))

                parset = np.array(
                    (
                        k1ap,
                        k1am,
                        k1bp,
                        k1bm,
                        k3ap,
                        k3am,
                        k3bp,
                        k3bm,
                        qx,
                        d1,
                        d3,
                        r10,
                        r20,
                        s10,
                        s30,
                        r1p,
                        r1m,
                        r2p,
                        r2m,
                        beta,
                        0,
                    )
                )  # Parameter set to feed into the integrator.
                R0 = np.array(
                    [
                        r10,
                        r20,
                        2,
                        0,
                        0,
                        s10,
                        s30,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                )  # Initial concentrations.
                R = integrate.odeint(
                    dIL27_dt, R0, tt, args=(parset,)
                )  # Integrate the mathematical model.
                (
                    n1,
                    n2,
                    n3,
                    n4,
                    n5,
                    n6,
                    n7,
                    n8,
                    n9,
                    n10,
                    n11,
                    n12,
                    n13,
                    n14,
                    n15,
                    n16,
                    n17,
                    n18,
                    n19,
                    n20,
                    n21,
                    n22,
                    n23,
                    n24,
                    n25,
                    n26,
                    n27,
                    n28,
                    n29,
                    n30,
                    n31,
                    n32,
                    n33,
                ) = R.T  # Model outputs for each vairable.
                pS1 = (
                    n12 + n13 + n17 + n18 + 2 * n19 + n26 + n29 + n30 + n31 + n32
                )  # Sum of variables containing pSTAT1.
                pS3 = (
                    n14 + n15 + n21 + n22 + 2 * n23 + n27 + n28 + n30 + n31 + n33
                )  # Sum of variables containing pSTAT3.
            # Add model simulations for total pSTAT1 and pSTAT3 to outputs list.
            outputs.append(
                np.array((pS1[0::100], pS3[0::100]))
            )  # Append model outputs (cut to length 181) to the list.

        paramsa.append(
            np.array(
                (
                    k1ap,
                    k1am,
                    k1bp,
                    k1bm,
                    k3ap,
                    k3am,
                    k3bp,
                    k3bm,
                    qx,
                    d1,
                    d3,
                    r10,
                    r20,
                    s10,
                    s30,
                )
            )
        )
        paramsa = np.hstack(
            (paramsa)
        )  # Create an array of the sampled parameter values.

        # Take the values of the model simulations that correlate to the data time points.
        time_points = np.array((0, 5, 15, 30, 60, 90, 120, 180))  # Data time points.
        IL6_pS1_model = np.empty((8,))
        IL6_pS3_model = np.empty((8,))
        IL27_pS1_model = np.empty((8,))
        IL27_pS3_model = np.empty((8,))
        for i in range(8):
            IL6_pS1_model[i] = outputs[0][0, :][time_points[i]]
            IL6_pS3_model[i] = outputs[0][1, :][time_points[i]]
            IL27_pS1_model[i] = outputs[1][0, :][time_points[i]]
            IL27_pS3_model[i] = outputs[1][1, :][time_points[i]]

        # Normalise the model in the same way as the data by dividing by IL-27 time 30 minutes.
        norm_point_s1 = IL27_pS1_model[3]
        norm_point_s3 = IL27_pS3_model[3]
        IL6_pS1_model_norm = IL6_pS1_model / norm_point_s1
        IL6_pS3_model_norm = IL6_pS3_model / norm_point_s3
        IL27_pS1_model_norm = IL27_pS1_model / norm_point_s1
        IL27_pS3_model_norm = IL27_pS3_model / norm_point_s3

        # Compute the distance between the model and the data.
        IL6_pS1_dis = []
        IL6_pS3_dis = []
        IL27_pS1_dis = []
        IL27_pS3_dis = []
        for t in range(8):
            IL6_pS1_dis.append(pow((IL6_pS1_model_norm[t] - IL6_final[0][t]), 2))
            IL6_pS3_dis.append(pow((IL6_pS3_model_norm[t] - IL6_final[1][t]), 2))
            IL27_pS1_dis.append(pow((IL27_pS1_model_norm[t] - IL27_final[0][t]), 2))
            IL27_pS3_dis.append(pow((IL27_pS3_model_norm[t] - IL27_final[1][t]), 2))

        dist = np.sum(
            (
                np.array(
                    (
                        np.sum(IL6_pS1_dis),
                        np.sum(IL6_pS3_dis),
                        np.sum(IL27_pS1_dis),
                        np.sum(IL27_pS3_dis),
                    )
                )
            )
        )
        dist2 = np.sqrt(dist)

        # If the distance is less than epsilon, append distance and parameters to their respective lists and compute the weight
        # for the accepted parameter set.
        if dist2 < epsilon:
            number += 1
            denom_arr = []
            for j in range(N):
                weight = ABC_runs[it][j, 26]
                params_row = ABC_runs[it][j, 1:26]
                boxs_up = []
                boxs_low = []
                for i in range(25):
                    if i in [21, 22, 23, 24]:
                        boxs_up.append(params_row[i] + sigma[i])
                        boxs_low.append(params_row[i] - sigma[i])
                    else:
                        boxs_up.append(np.log10(params_row[i]) + sigma[i])
                        boxs_low.append(np.log10(params_row[i]) - sigma[i])
                outside = 0
                for i in range(25):
                    if i in [21, 22, 23, 24]:
                        if parameters[i] < boxs_low[i] or parameters[i] > boxs_up[i]:
                            outside = 1
                    else:
                        if (
                            np.log10(parameters[i]) < boxs_low[i]
                            or np.log10(parameters[i]) > boxs_up[i]
                        ):
                            outside = 1
                if outside == 1:
                    denom_arr.append(0)
                else:
                    denom_arr.append(weight * np.prod(1 / (2 * sigma)))

            weight_param = 1 / np.sum(denom_arr)
            weights_hyp1 = np.hstack((weights_hyp1, weight_param))
            results_hyp1 = np.hstack((results_hyp1, dist2))
            accepted_params_hyp1 = np.vstack((accepted_params_hyp1, paramsa))
            priors_hyp1 = np.vstack((priors_hyp1, prior_sample))

    # Normalise the weights
    weights_hyp1_2 = weights_hyp1 / np.sum(weights_hyp1)
    weights_hyp1_3 = np.reshape(weights_hyp1_2, (len(weights_hyp1_2), 1))

    # Print information about the first run of the ABC.
    print("Acceptance rate for iteration " + str(it + 1) + ": " + str(N * 100 / truns))
    print("Epsilon = " + str(epsilon))
    print("Total runs = " + str(truns))
    # Return the results (distances, accepted parameters, weights, total number of runs).
    return [
        np.hstack(
            (
                np.reshape(results_hyp1, (len(accepted_params_hyp1), 1)),
                accepted_params_hyp1,
                weights_hyp1_3,
            )
        ),
        truns,
    ]


start = time.time()
# Sample size for the ABC-SMC
N = 10000
# Set the array of epsilon values to be used.
epsilons = [10, 5, 3, 2.5, 2.25, 2, 1.75]
# Run the first iteration with a sufficiently large value of epsilon.
first = first_iteration(N, 100)
ABC_runs = []
ABC_runs.append(first[0])
# Run the sucessive iterations of the ABC.
for itt in range(len(epsilons)):
    run = other_iterations(N, itt)
    ABC_runs.append(run[0])
# Save the results as a text file.
print(f"baseline run took {time.time() - start} seconds")
np.savetxt("TH1_posteriors.txt", ABC_runs[itt + 1])
