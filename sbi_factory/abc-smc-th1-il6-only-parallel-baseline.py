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
import diffrax
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import equinox as eqx

# Read in the experimental data.
IL6_data_pS1 = np.loadtxt(f"IL6_data_pS1_mean_TH1.txt")
IL6_data_pS3 = np.loadtxt(f"IL6_data_pS3_mean_TH1.txt")
# IL6_final = [IL6_data_pS1 / IL6_data_pS1[3],IL6_data_pS3 / IL6_data_pS3[3]] #renormalize to IL6
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
            gamma,
        ) = self.params

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
    accepted_params_hyp1 = np.empty((0, 16))
    print("Running iteration: 0")

    # Function to run batched simulation
    def generate_batch_simulation(batch_size):
        # np.random.seed(0)
        setx = np.random.randint(0, len(RPE_posteriors), size=batch_size)
        k1ap = RPE_posteriors[setx, 10]
        k1am = RPE_posteriors[setx, 11]
        k1bp = RPE_posteriors[setx, 12]
        k1bm = RPE_posteriors[setx, 13]
        k3ap = RPE_posteriors[setx, 14]
        k3am = RPE_posteriors[setx, 15]
        k3bp = RPE_posteriors[setx, 16]
        k3bm = RPE_posteriors[setx, 17]
        qx = pow(2, np.random.uniform(-3, 2, batch_size))
        d1 = pow(2, np.random.uniform(-5, -2, batch_size))
        d3 = pow(2, np.random.uniform(-5, -2, batch_size))
        r10 = np.random.normal(12.7, 6.35, batch_size)
        r20 = np.random.normal(33.8, 16.9, batch_size)
        s10 = np.random.normal(300, 100, batch_size)
        s30 = np.random.normal(400, 100, batch_size)
        r1p = RPE_posteriors[setx, 0]
        r1m = RPE_posteriors[setx, 1]
        r2p = RPE_posteriors[setx, 2]
        r2m = RPE_posteriors[setx, 3]
        beta = np.power(2, np.random.uniform(-5, -1, batch_size))
        gamma = np.zeros(batch_size)

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
                gamma,
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

    # def compute_distances(R_IL6):

    #     # Convert results back to NumPy for post-processing
    #     R_IL6 = np.array(R_IL6)

    #     # Extract relevant outputs for each model
    #     pS1_IL6 = R_IL6[:, :, 8] + R_IL6[:, :, 11] + 2 * R_IL6[:, :, 12] + R_IL6[:, :, 17] + R_IL6[:, :, 19] + R_IL6[:, :, 20]
    #     pS3_IL6 = R_IL6[:, :, 9] + R_IL6[:, :, 14] + 2 * R_IL6[:, :, 15] + R_IL6[:, :, 18] + R_IL6[:, :, 19] + R_IL6[:, :, 21]

    #     # Time points for data
    #     time_points = np.array([0, 5, 15, 30, 60, 90, 120, 180])

    #     pS1_IL27_norm = 1 # pS1_IL27[time_points[4]]
    #     pS3_IL27_norm = 1 # pS3_IL27[time_points[4]]

    #     IL6_pS1_dis = np.sum(np.power(pS1_IL6[:,time_points] / pS1_IL27_norm - IL6_final[0][np.newaxis, :], 2), axis=1)
    #     IL6_pS3_dis = np.sum(np.power(pS3_IL6[:,time_points] / pS3_IL27_norm - IL6_final[1][np.newaxis, :], 2), axis=1)

    #     # IL6_pS1_dis = np.sum(np.power(pS1_IL6[time_points] / pS1_IL27_norm - IL6_final[0][:, np.newaxis], 2), axis=0)
    #     # IL6_pS3_dis = np.sum(np.power(pS3_IL6[time_points] / pS3_IL27_norm - IL6_final[1][:, np.newaxis], 2), axis=0)

    #     distances = np.sqrt(IL6_pS1_dis + IL6_pS3_dis)
    #     return distances
    while accepted_params_hyp1.shape[0] < N:
        # Execute the batched simulation
        R0_IL6, params_batch_IL6, tt = generate_batch_simulation(batch_size)

        distances = solve_ode_batch(R0_IL6, tt, params_batch_IL6)

        # distances = compute_distances(R_IL6)
        accepted_params = params_batch_IL6[distances < epsilon]
        # append accepted_params to accepted_param_hyp1
        accepted_params_hyp1 = np.vstack(
            (accepted_params_hyp1, np.array(accepted_params))
        )

    return accepted_params_hyp1


def other_iterations_batched(N, it, batch_size):
    print("Running iteration: " + str(it + 1))
    epsilon = epsilons[it]
    p_list = [i for i in range(N)]

    lower_bounds = [-10, -15, -5, -7, -2, -7, -2, -7, -2, -3, -5, -5, 0, 0, 0, 0]
    upper_bounds = [5, 5, -1, 1, 1, 1, 1, 1, 1, 2, -2, -2, 40, 100, 800, 1000]

    ranges = []
    for i in range(16):
        # print("shape of params to figure out where the batch is",ABC_runs[it].shape)
        if i in [2, 9, 11, 12, 13, 14, 15]:
            if i in [12, 13, 14, 15]:
                r1 = np.max(ABC_runs[it][:, i]) - np.min(ABC_runs[it][:, i])
            else:
                r1 = np.max(
                    np.log10(np.clip(ABC_runs[it][:, i], a_min=1e-9, a_max=1e9))
                ) - np.min(np.log10(np.clip(ABC_runs[it][:, i], a_min=1e-9, a_max=1e9)))
            ranges.append(r1)
        else:
            r1 = upper_bounds[i] - lower_bounds[i]
            ranges.append(r1)

    ranges_arr = np.asarray(ranges)
    sigma = [
        (0.2 * ranges_arr[i]) if i in [2, 9, 11, 12, 13, 14, 15] else ranges_arr[i]
        for i in range(16)
    ]
    sigma = np.asarray(sigma)
    # print("sigma",sigma)

    priors_hyp1 = np.empty((0, 16))
    accepted_params_hyp1 = np.empty((0, 16))
    results_hyp1 = np.empty((0))
    weights_hyp1 = np.empty((0))

    number = 0
    truns = 0

    while number < N:
        truns += 1

        batch_indices = np.random.choice(p_list, batch_size)  # , p=ABC_runs[it][:, 15])
        prior_samples = ABC_runs[it][batch_indices][:, range(16)]

        parameters_batch = []
        for i in range(16):
            prior_samples_i = prior_samples[:, i]
            prior_samples_i = prior_samples_i[~np.isnan(prior_samples_i)]
            if i in [2, 9, 11, 12, 13, 14, 15]:
                if i in [12, 13, 14, 15]:
                    lower = np.clip(
                        np.min(prior_samples_i - sigma[i]), a_min=-1e9, a_max=1e9
                    )
                    upper = np.clip(
                        np.max(prior_samples_i + sigma[i]), a_min=-1e9, a_max=1e9
                    )
                else:
                    lower = np.clip(
                        np.log10(np.clip(prior_samples_i, a_min=1e-9, a_max=1e9))
                        - sigma[i],
                        a_min=-1e9,
                        a_max=1e9,
                    )
                    upper = np.clip(
                        np.log10(np.clip(prior_samples_i, a_min=1e-9, a_max=1e9))
                        + sigma[i],
                        a_min=-1e9,
                        a_max=1e9,
                    )
                # print(i, lower, upper)
                parameter = np.random.uniform(low=lower, high=upper, size=batch_size)
                if i in [12, 13, 14, 15]:
                    parameters_batch.append(parameter)
                else:
                    parameters_batch.append(10**parameter)
            else:
                parameter = np.random.choice(RPE_posteriors[:, i], batch_size)
                parameters_batch.append(parameter)

        parameters_batch = np.vstack(parameters_batch).T

        # feasible_mask = np.all([(parameters_batch[:, ik] >= lower_bounds[ik]) & (parameters_batch[:, ik] <= upper_bounds[ik])
        #                         if ik in [12,13,14,15]
        #                         else (parameters_batch[:, ik] >= 10**lower_bounds[ik]) & (parameters_batch[:, ik] <= 10**upper_bounds[ik])
        #                         for ik in range(16)], axis=0)

        # feasible_indices = np.where(feasible_mask)[0]
        # print(feasible_indices)

        # if len(feasible_indices) > 0:
        #     number += len(feasible_indices)

        #     feasible_parameters = parameters_batch[feasible_indices]

        #     k1ap, k1am, k3ap, k3am, qx, d1, d3, r10, s10, s30, r1p, r1m, r2p, r2m, beta, gamma = feasible_parameters.T

        #     parset = np.column_stack((k1ap, k1am, k3ap, k3am, qx, d1, d3, r10, s10, s30, r1p, r1m, r2p, r2m, beta, np.zeros(len(feasible_indices))))
        #     R0 = np.column_stack((r10, np.full(len(feasible_indices), 10), np.zeros((len(feasible_indices), 16))))

        # Initial conditions
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

        dist = solve_ode_batch(R0, tt, parameters_batch)

        accepted = dist < epsilon
        accepted_parameters = parameters_batch[accepted]

        if accepted_parameters.shape[0] > 0:
            accepted_count = accepted_parameters.shape[0]
            number += accepted_count

            denom_arr = np.zeros(accepted_count)
            for j in range(N):
                weight = np.ones_like(ABC_runs[it][j, 15])
                params_row = ABC_runs[it][j]
                boxs_up = np.array(
                    [
                        (
                            params_row[i] + sigma[i]
                            if i in [12, 13, 14, 15]
                            else np.log10(params_row[i]) + sigma[i]
                        )
                        for i in range(16)
                    ]
                )
                boxs_low = np.array(
                    [
                        (
                            params_row[i] - sigma[i]
                            if i in [12, 13, 14, 15]
                            else np.log10(params_row[i]) - sigma[i]
                        )
                        for i in range(16)
                    ]
                )

                outside = np.any(
                    (accepted_parameters < boxs_low) | (accepted_parameters > boxs_up),
                    axis=1,
                )
                denom_arr += np.where(outside, 0, weight * np.prod(1 / (2 * sigma)))

            weight_params = 1 / denom_arr
            weights_hyp1 = np.hstack((weights_hyp1, weight_params))
            results_hyp1 = np.hstack((results_hyp1, dist[accepted]))
            accepted_params_hyp1 = np.vstack(
                (accepted_params_hyp1, accepted_parameters)
            )
            print(f"\rAccepted : {accepted_params_hyp1.shape[0]}/{N}", end="")
            priors_hyp1 = np.vstack((priors_hyp1, prior_samples[:][accepted]))

    weights_hyp1_2 = weights_hyp1 / np.sum(weights_hyp1)
    weights_hyp1_3 = np.reshape(weights_hyp1_2, (len(weights_hyp1_2), 1))

    print(
        "Acceptance rate for iteration "
        + str(it + 1)
        + ": "
        + str(N * 100 / (truns * batch_size))
    )
    print("Epsilon = " + str(epsilon))
    print("Total runs = " + str(truns))

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


# dry run
first_iteration_parallel(1000, 100, 1000)
start = time.time()
N = 10000
batch_size = 10000
ABC_runs = []
ABC_runs.append(first_iteration_parallel(N, 100, batch_size))
print(time.time() - start)
ABC_runs[0] = ABC_runs[0][:N, 0:16]

epsilons = [10, 5, 3, 2.5, 2.25, 2, 1.75]
for it in range(len(epsilons)):
    ABC_runs.append(other_iterations_batched(N, it, batch_size))
    ABC_runs[it + 1] = ABC_runs[0][:N, 0:16]

print(time.time() - start)
