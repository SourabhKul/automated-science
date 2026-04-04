import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.experimental.ode import odeint
import numpy as np
import time
from jax.lib import xla_bridge

def ode_system(R, t, params):
    k1ap, k1am, k1bp, k1bm, k3ap, k3am, k3bp, k3bm, qx, d1, d3, r10, r20, s10, s30, r1p, r1m, r2p, r2m, beta, gamma = params

    dRdt = jnp.array([
        -r1p*R[0]*R[1] + r1m*R[2] - beta*R[0] - gamma*(R[20]+R[21])*R[0],
        -r1p*R[0]*R[1] + r1m*R[2],
        r1p*R[0]*R[1] - r1m*R[2] - 2*r2p*R[2]**2 + 2*r2m*R[3] - beta*R[2] - gamma*(R[20]+R[21])*R[2],
        r2p*R[2]**2 - r2m*R[3] - 2*k1ap*R[3]*R[4] + k1am*R[6] - 2*k3ap*R[3]*R[5] + k3am*R[7] + k1am*R[8] + k3am*R[9] - beta*R[3] - gamma*(R[20]+R[21])*R[3],
        -k1ap*R[4]*(R[6]+R[7]+R[8]+R[9]) + k1am*(2*R[10]+R[16]+R[11]+R[18]) - 2*k1ap*R[3]*R[4] + k1am*R[6] + d1*R[20],
        -k3ap*R[5]*(R[6]+R[7]+R[8]+R[9]) + k3am*(2*R[13]+R[16]+R[14]+R[17]) - 2*k3ap*R[3]*R[5] + k3am*R[7] + d3*R[21],
        2*k1ap*R[3]*R[4] - k1am*R[6] - k1ap*R[6]*R[4] + 2*k1am*R[10] - k3ap*R[6]*R[5] + k3am*R[16] - qx*R[6] + k1am*R[11] + k3am*R[18] - beta*R[6] - gamma*(R[20]+R[21])*R[6],
        2*k3ap*R[3]*R[5] - k3am*R[7] - k3ap*R[7]*R[5] + 2*k3am*R[13] - k1ap*R[7]*R[4] + k1am*R[16] - qx*R[7] + k3am*R[14] + k1am*R[17] - beta*R[7] - gamma*(R[20]+R[21])*R[7],
        -k1ap*R[4]*R[8] + k1am*R[11] - k3ap*R[5]*R[8] + k3am*R[17] + qx*R[6] - k1am*R[8] + 2*k1am*R[12] + k3am*R[19] - beta*R[8] - gamma*(R[20]+R[21])*R[8],
        -k3ap*R[5]*R[9] + k3am*R[14] - k1ap*R[4]*R[9] + k1am*R[18] + qx*R[7] - k3am*R[9] + 2*k3am*R[15] + k1am*R[19] - beta*R[9] - gamma*(R[20]+R[21])*R[9],
        k1ap*R[4]*R[6] - 2*k1am*R[10] - 2*qx*R[10] - beta*R[10] - gamma*(R[20]+R[21])*R[10],
        k1ap*R[8]*R[4] - k1am*R[11] + 2*qx*R[10] - (qx + k1am)*R[11] - beta*R[11] - gamma*(R[20]+R[21])*R[11],
        qx*R[11] - 2*k1am*R[12] - beta*R[12] - gamma*(R[20]+R[21])*R[12],
        k3ap*R[5]*R[7] - 2*k3am*R[13] - 2*qx*R[13] - beta*R[13] - gamma*(R[20]+R[21])*R[13],
        k3ap*R[9]*R[5] - k3am*R[14] + 2*qx*R[13] - (qx + k3am)*R[14] - beta*R[14] - gamma*(R[20]+R[21])*R[14],
        qx*R[14] - 2*k3am*R[15] - beta*R[15] - gamma*(R[20]+R[21])*R[15],
        k1ap*R[4]*R[7] - k1am*R[16] + k3ap*R[6]*R[5] - k3am*R[16] - 2*qx*R[16] - beta*R[16] - gamma*(R[20]+R[21])*R[16],
        qx*R[16] + k3ap*R[8]*R[5] - k3am*R[17] - qx*R[17] - k1am*R[17] - beta*R[17] - gamma*(R[20]+R[21])*R[17],
        qx*R[16] + k1ap*R[9]*R[4] - k1am*R[18] - qx*R[18] - k3am*R[18] - beta*R[18] - gamma*(R[20]+R[21])*R[18],
        qx*R[18] + qx*R[17] - k1am*R[19] - k3am*R[19] - beta*R[19] - gamma*(R[20]+R[21])*R[19],
        k1am*(R[8]+R[11]+R[17]+R[19]) + 2*k1am*R[12] - d1*R[20],
        k3am*(R[9]+R[14]+R[18]+R[19]) + 2*k3am*R[15] - d3*R[21],
    ])
    return dRdt

# Vectorize the ODE system
v_ode_system = vmap(ode_system, in_axes=(0, None, 0))

@jit
def solve_ode_batch(y0, t, params):
    solution = odeint(v_ode_system, y0, t, params)
    return solution

def generate_random_params(num_instances):
    # Generate random parameters for the ODE system
    return np.random.rand(num_instances, 21).astype(np.float32)

def generate_initial_conditions(num_instances, dim):
    # Generate random initial conditions
    return np.random.rand(num_instances, dim).astype(np.float32)

# dry run to compile
params = generate_random_params(10)
initial_conditions = generate_initial_conditions(10, 22)
time_points = np.linspace(0.0, 10800, 18001) # np.linspace(0.0, 10.0, num=100).astype(np.float32)

# Convert to JAX arrays
params_jax = jnp.array(params)
initial_conditions_jax = jnp.array(initial_conditions)
time_points_jax = jnp.array(time_points)

# Solve the ODE for the batch
solution = solve_ode_batch(initial_conditions_jax, time_points_jax, params_jax)

def run_e2e(times):
    for num_instances in times:
        start_time = time.time()
        # Generate random parameters and initial conditions
        params = generate_random_params(num_instances)
        initial_conditions = generate_initial_conditions(num_instances, 22)
        time_points = np.linspace(0.0, 10.0, num=100).astype(np.float32)

        # Convert to JAX arrays
        params_jax = jnp.array(params)
        initial_conditions_jax = jnp.array(initial_conditions)
        time_points_jax = jnp.array(time_points)

        # Solve the ODE for the batch
        solutions = solve_ode_batch(initial_conditions_jax, time_points_jax, params_jax)


        print(f"Total time for {num_instances} instances was {time.time() - start_time} seconds")
        return solutions


solutions = run_e2e([30000]) # 
print(solutions[180][34][:])
