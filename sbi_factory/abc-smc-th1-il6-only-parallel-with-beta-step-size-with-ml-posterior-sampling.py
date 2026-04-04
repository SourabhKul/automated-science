import numpy as np
from scipy import integrate
import time
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.experimental.ode import odeint
import numpy as np
import time
from jax.lib import xla_bridge
from jax import random
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import equinox as eqx
import pickle
import tensorflow as tf
import xgboost as xgb
from tensorflow.keras.metrics import MeanAbsolutePercentageError

nn_ps = tf.keras.models.load_model(
    "trained_model_with_mape_loss.h5",
    custom_objects={"mape": MeanAbsolutePercentageError},
)

xgb_ps = xgb.XGBRegressor()

xgb_ps.load_model("trained_model_with_xgboost.json")

ps_method = "nn"

# jax.config.update("jax_platform_name", "cpu")

# Read in the experimental data.
IL6_data_pS1 = np.loadtxt(f"IL6_data_pS1_mean_TH1.txt")
IL6_data_pS3 = np.loadtxt(f"IL6_data_pS3_mean_TH1.txt")
# IL6_final = [
#     IL6_data_pS1 / IL6_data_pS1[3],
#     IL6_data_pS3 / IL6_data_pS3[3],
# ]  # renormalize to IL6
IL6_final = [IL6_data_pS1, IL6_data_pS3]
RPE_posteriors = np.loadtxt(f"RPE1_posteriors.txt")


class MyODESystem(eqx.Module):
    params: np.array

    def __init__(self, params):
        self.params = params  # Store params dictionary

    def __call__(self, t, R, args):
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
        ) = self.params
        gamma = 0.0
        return jnp.clip(
            jnp.array(
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
                    k1am * (R[8] + R[11] + R[17] + R[19])
                    + 2 * k1am * R[12]
                    - d1 * R[20],
                    k3am * (R[9] + R[14] + R[18] + R[19])
                    + 2 * k3am * R[15]
                    - d3 * R[21],
                ]
            ),
            -1e2,
            1e2,
        )


# Set the timecourse for the integration (units of seconds).
tt = np.linspace(0.0, 108, 180)  # np.linspace(0.0, 10800, 18001)


# Function to solve the ODE using Diffrax
@jit
def solve_ode_batch(y0, t, params):
    solver = Tsit5()
    saveat = SaveAt(ts=t)

    @jit
    def solve_single(y0, params):
        ode_system = MyODESystem(params)
        term = ODETerm(ode_system)
        sol = diffeqsolve(
            term, solver, t0=t[0], t1=t[-1], dt0=0.1, y0=y0, saveat=saveat
        )
        return sol.ys

    vmap_solve = jax.vmap(solve_single, in_axes=(0, 0))

    R_IL6 = vmap_solve(y0, params)

    # Extract relevant outputs for each model
    pS1_IL6 = (
        R_IL6[:, :, 8]
        + R_IL6[:, :, 11]
        + 2 * R_IL6[:, :, 12]
        + R_IL6[:, :, 17]
        + R_IL6[:, :, 19]
        + R_IL6[:, :, 20]
    )
    pS3_IL6 = (
        R_IL6[:, :, 9]
        + R_IL6[:, :, 14]
        + 2 * R_IL6[:, :, 15]
        + R_IL6[:, :, 18]
        + R_IL6[:, :, 19]
        + R_IL6[:, :, 21]
    )

    # Time points for data
    time_points = jnp.array([0, 5, 15, 30, 60, 90, 120, 180])

    pS1_IL27_norm = 1  # pS1_IL27[time_points[4]]
    pS3_IL27_norm = 1  # pS3_IL27[time_points[4]]

    IL6_pS1_dis = jnp.sum(
        jnp.power(
            pS1_IL6[:, time_points] / pS1_IL27_norm - IL6_final[0][jnp.newaxis, :], 2
        ),
        axis=1,
    )
    IL6_pS3_dis = jnp.sum(
        jnp.power(
            pS3_IL6[:, time_points] / pS3_IL27_norm - IL6_final[1][jnp.newaxis, :], 2
        ),
        axis=1,
    )

    # IL6_pS1_dis = np.sum(np.power(pS1_IL6[time_points] / pS1_IL27_norm - IL6_final[0][:, np.newaxis], 2), axis=0)
    # IL6_pS3_dis = np.sum(np.power(pS3_IL6[time_points] / pS3_IL27_norm - IL6_final[1][:, np.newaxis], 2), axis=0)

    distances = jnp.sqrt(IL6_pS1_dis + IL6_pS3_dis)

    return distances


# Function to run the first iteration of the hypothesis selection. A hypothesis is chosen by at random and the parameters for
# this hypothesis are sampled from their prior distributions. The distance is computed between the model and the data and stored in a vector.
# This process is repeated until there are N accepted parameters sets from hypothesis 1 and 2 combined.
def first_iteration_parallel(N, epsilon, batch_size):
    accepted_params_hyp1 = np.empty((0, 15))
    accepted_distances = np.empty(0)
    print("Running iteration: 0")

    # Function to run batched simulation
    def generate_batch_simulation(batch_size):
        # np.random.seed(0)

        k1ap = np.power(10, np.random.uniform(-7, 1, batch_size))
        k1am = np.power(10, np.random.uniform(-2, 1, batch_size))
        k3ap = np.power(10, np.random.uniform(-7, 1, batch_size))
        k3am = np.power(10, np.random.uniform(-2, 1, batch_size))
        qx = np.power(10, np.random.uniform(-3, 2, batch_size))
        d1 = np.power(10, np.random.uniform(-5, -2, batch_size))
        d3 = np.power(10, np.random.uniform(-5, -2, batch_size))
        r10 = 12.7 + 6.35 * np.random.normal(size=batch_size)
        s10 = 300 + 100 * np.random.normal(size=batch_size)
        s30 = 400 + 100 * np.random.normal(size=batch_size)
        r2p = np.power(10, np.random.uniform(-2, 3, batch_size))
        r2m = np.power(10, np.random.uniform(-3, 1, batch_size))
        r1p = np.power(10, -3 + 1.5 * np.random.normal(size=batch_size))
        r1m = np.power(10, -3.9 + 1.96 * np.random.normal(size=batch_size))
        beta = np.power(2, np.random.uniform(-5, -1, batch_size))

        # setx = np.random.randint(0, len(RPE_posteriors), size=batch_size)
        # k1ap = RPE_posteriors[setx, 10]
        # k1am = RPE_posteriors[setx, 11]
        # k3ap = RPE_posteriors[setx, 14]
        # k3am = RPE_posteriors[setx, 15]
        # qx = pow(2, np.random.uniform(-3, 2, batch_size))
        # d1 = pow(2, np.random.uniform(-5, -2, batch_size))
        # d3 = pow(2, np.random.uniform(-5, -2, batch_size))
        # r10 = np.random.normal(12.7, 6.35, batch_size)
        # s10 = np.random.normal(300, 100, batch_size)
        # s30 = np.random.normal(400, 100, batch_size)
        # r1p = RPE_posteriors[setx, 0]
        # r1m = RPE_posteriors[setx, 1]
        # r2p = RPE_posteriors[setx, 2]
        # r2m = RPE_posteriors[setx, 3]
        # beta = np.power(2, np.random.uniform(-5, -1, batch_size))

        # Combine parameters into a batched parameter set
        params_batch_IL6 = np.stack(
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
            ),
            axis=-1,
        )

        # Initial conditions
        zeros = np.zeros(batch_size)
        R0_IL6 = np.stack(
            [
                r10,
                np.full(batch_size, 10),
                zeros,
                zeros,
                s10,
                s30,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
            ],
            axis=-1,
        )
        # Convert to JAX arrays for ODE solver
        R0_IL6 = jnp.array(R0_IL6)
        params_batch_IL6 = jnp.array(params_batch_IL6)

        tt = jnp.linspace(0.0, 108, 181)  # jnp.linspace(0.0, 108, 181)

        return R0_IL6, params_batch_IL6, tt

    while accepted_params_hyp1.shape[0] < N:
        print(f"Progress: {100 * accepted_params_hyp1.shape[0] / N}%", end="\r")

        # Execute the batched simulation
        R0_IL6, params_batch_IL6, tt = generate_batch_simulation(batch_size)

        distances = solve_ode_batch(R0_IL6, tt, params_batch_IL6)

        # distances = compute_distances(R_IL6)
        accepted_params = params_batch_IL6[distances < epsilon]
        # append accepted_params to accepted_param_hyp1
        accepted_params_hyp1 = np.vstack(
            (accepted_params_hyp1, np.array(accepted_params))
        )
        accepted_distances = np.hstack(
            (accepted_distances, distances[distances < epsilon])
        )
    assert (
        accepted_params_hyp1.shape[0] == accepted_distances.shape[0]
    ), f"Mismatched shapes params: {accepted_params_hyp1.shape}, distances: {accepted_distances.shape}"
    return {
        "params": accepted_params_hyp1,
        "distances": accepted_distances,
    }


def other_iterations_batched(N, it, batch_size):

    crossover = 0
    N = 5 * N if it < crossover else N

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    print("Running iteration: " + str(it + 1))
    epsilon = epsilons[it]

    # Upper and lower bounds for the uniform distributions of the priors for each parameter.
    lower_bounds = [
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
        -10,
        -15,
        -2,
        -3,
        -5,
    ]
    upper_bounds = [
        1,
        1,
        1,
        1,
        2,
        -2,
        -2,
        40,
        800,
        1000,
        5,
        5,
        3,
        1,
        -1,
    ]

    ranges = []
    for i in range(15):
        # print("shape of params to figure out where the batch is",ABC_runs[it].shape)
        if i in [4, 5, 6, 7, 8, 9, 14]:
            if i in [6, 7, 8, 9]:
                r1 = np.max(ABC_runs[it]["params"][:, i]) - np.min(
                    ABC_runs[it]["params"][:, i]
                )
            else:
                r1 = np.max(
                    np.log10(
                        np.clip(ABC_runs[it]["params"][:, i], a_min=1e-9, a_max=1e9)
                    )
                ) - np.min(
                    np.log10(
                        np.clip(ABC_runs[it]["params"][:, i], a_min=1e-9, a_max=1e9)
                    )
                )
            ranges.append(r1)
        else:
            r1 = upper_bounds[i] - lower_bounds[i]
            ranges.append(r1)

    ranges_arr = np.asarray(ranges)
    sigma = [
        (
            (0.2 * ranges_arr[i] + 1e-9)
            if i in [4, 5, 6, 7, 8, 9, 14]
            else ranges_arr[i] + 1e-9
        )
        for i in range(15)
    ]
    sigma = np.asarray(sigma)

    priors_hyp1 = np.empty((0, 15))
    accepted_params_hyp1 = np.empty((0, 15))
    results_hyp1 = np.empty((0))
    weights_hyp1 = np.empty((0))
    cov_matrix = jnp.cov(ABC_runs[it]["params"].T)
    cov_diag = jnp.diag(
        cov_matrix
    )  # Get the diagonal elements of the covariance matrix
    number = 0
    truns = 0
    accepted_distances = np.empty((0))
    accepted_count = 2000
    while number < N:
        truns += 1

        if it < crossover:
            batch_size = 500 * batch_size
        # Sample previous stage parameters randomly
        batch_indices = np.random.choice(
            np.arange(ABC_runs[it]["params"].shape[0]),
            batch_size,
            replace=True,
            p=softmax(-ABC_runs[it]["distances"]),
        )
        prior_samples = ABC_runs[it]["params"][batch_indices][:, range(15)]

        # Initialize a list to store the parameter batches
        parameters_batch = []

        # Function to sample Beta-distributed step sizes for the entire batch
        def sample_beta_step_size(key, batch_size, alpha, beta_val):
            return random.beta(key, alpha, beta_val, shape=(batch_size,))

        # Sample sigma value from a Beta distribution for each parameter in the batch
        prev_run_acceptance_rate = accepted_count / batch_size
        alpha = 10 * prev_run_acceptance_rate
        beta_val = 2.0 + it  #  * (epsilons[it] / epsilons[it - 1])
        rng = jax.random.PRNGKey(43)
        rng, key = random.split(rng)
        sigma_beta = sample_beta_step_size(key, batch_size, alpha, beta_val)
        for i in range(15):
            prior_samples_i = prior_samples[:, i]
            assert np.sum(np.isnan(prior_samples_i)) == 0
            if i in [4, 5, 6, 7, 8, 9, 14]:
                if i in [6, 7, 8, 9]:
                    lower = np.clip(
                        prior_samples_i - sigma[i],
                        a_min=-1e9,
                        a_max=1e9,
                    )
                    upper = np.clip(
                        prior_samples_i + sigma[i],
                        a_min=-1e9,
                        a_max=1e9,
                    )
                else:
                    lower = np.clip(
                        np.log10(np.clip(prior_samples_i, a_min=1e-9, a_max=1e9))
                        - sigma[i],
                        a_min=-9,
                        a_max=9,
                    )
                    upper = np.clip(
                        np.log10(np.clip(prior_samples_i, a_min=1e-9, a_max=1e9))
                        + sigma[i],
                        a_min=-9,
                        a_max=9,
                    )
                assert np.all(
                    lower < upper
                ), f"For parameter {i}, at {np.count_nonzero(lower < upper)} indices"
                prior_samples_i = np.clip(prior_samples_i, a_min=lower, a_max=upper)

                # Apply Gaussian perturbation to each parameter in the batch using truncated normal sampling
                stddevs = sigma_beta * (upper - lower) + 0.2975 * np.clip(
                    jnp.sqrt(cov_diag[i]), a_min=1e-9, a_max=1e2
                )
                assert np.all(stddevs > 0)
                # print(i, np.mean(stddevs))

                # parameter = np.random.normal(prior_samples_i, stddevs)
                # # just normal, clip later:

                # parameter = np.clip(
                #     np.random.normal(prior_samples_i, stddevs), a_max=upper, a_min=lower
                # )

                # truncated normal

                rng, key = random.split(rng)
                a = (lower - prior_samples_i) / stddevs
                b = (upper - prior_samples_i) / stddevs
                parameter = (
                    random.truncated_normal(key, a, b) * stddevs + prior_samples_i
                )

                # # baseline

                # parameter = np.random.uniform(low=lower, high=upper, size=batch_size)

                assert parameter.shape[0] == batch_size

                if i in [6, 7, 8, 9]:
                    parameters_batch.append(parameter)
                else:
                    parameters_batch.append(10**parameter)
            else:
                index_remap = {0: 10, 1: 11, 2: 14, 3: 15, 10: 0, 11: 1, 12: 2, 13: 3}
                parameter = np.random.choice(
                    RPE_posteriors[:, index_remap[i]], batch_size
                )
                parameters_batch.append(parameter)

        # Stack the parameters batch to form the final array
        parameters_batch = np.vstack(parameters_batch).T
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
        ) = parameters_batch.T
        zeros = np.zeros(batch_size)
        R0 = np.stack(
            [
                r10,
                np.full(batch_size, 10),
                zeros,
                zeros,
                s10,
                s30,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
            ],
            axis=-1,
        )

        if it < crossover:
            dist = (
                np.array(nn_ps(parameters_batch)).reshape(-1)
                if ps_method == "nn"
                else np.array(xgb_ps.predict(parameters_batch)).reshape(-1)
            )

        else:
            dist = solve_ode_batch(R0, tt, parameters_batch)

        accepted = dist < epsilon
        accepted_parameters = parameters_batch[accepted]

        if accepted_parameters.shape[0] > 0:
            accepted_count = accepted_parameters.shape[0]
            accepted_distances = np.hstack((accepted_distances, dist[accepted]))
            number += accepted_count
            accepted_params_hyp1 = np.vstack(
                (accepted_params_hyp1, accepted_parameters)
            )
            print(f"\rAccepted : {accepted_params_hyp1.shape[0]}/{N}", end="")
            priors_hyp1 = np.vstack((priors_hyp1, prior_samples[:][accepted]))

    percentile = 50  # np.min([5 + 5 * (it), 50])
    next_epsilon = np.percentile(accepted_distances, percentile)
    if len(epsilons) == it + 1:
        epsilons.append(next_epsilon)
    else:
        epsilons[it + 1] = next_epsilon
    print(
        "Acceptance rate for iteration "
        + str(it + 1)
        + ": "
        + str(N * 100 / (truns * batch_size))
    )
    print("Epsilon = " + str(epsilon))
    print(f"Next Epsilon = {next_epsilon}")
    print("Total runs = " + str(truns))
    print(f"Time till this stage: {time.time() - start}")
    print(f"Percentile: {percentile}")
    print(np.percentile(accepted_distances, [1, 5, 10, 25, 50]))
    print("Mean of step size: ", np.mean(stddevs))

    return {
        "params": accepted_params_hyp1[accepted_distances < epsilons[it + 1]],
        "distances": accepted_distances[accepted_distances < epsilons[it + 1]],
    }

    # return {
    #     "params": accepted_params_hyp1,
    #     "distances": accepted_distances,
    # }


first_iteration_parallel(10, 100, 1000)
start = time.time()
N = 10000
batch_size = 100000
ABC_runs = {}
ABC_runs[0] = first_iteration_parallel(N, 10, batch_size)

print(time.time() - start)
# ABC_runs[0] = ABC_runs[0][:N, 0:15]
target_epsilon = 0.6
# Set the array of epsilon values to be used.
epsilons = [5, 5, 3, 2.5, 2.25, 2, 1.75, 1.5, 1.25, 1.1, 1, 0.9, 0.8, 0.7, 0.6]
it = 0
while True:
    ABC_runs[it + 1] = other_iterations_batched(N, it, batch_size)
    if epsilons[it] <= target_epsilon:
        print("final time:", time.time() - start)
        params = ABC_runs[it + 1]["params"]
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
        ) = params.T
        zeros = np.zeros(params.shape[0])
        batch_size = params.shape[0]
        R0 = np.stack(
            [
                r10,
                np.full(batch_size, 10),
                zeros,
                zeros,
                s10,
                s30,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
            ],
            axis=-1,
        )
        true_dist = solve_ode_batch(R0, tt, params)
        print(np.count_nonzero(true_dist < 1.75))
        break
    it = it + 1


print(time.time() - start)
