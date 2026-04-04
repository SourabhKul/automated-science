#A python program to run the ABC-SMC model selection for the RPE1 cell type.
#"Approximate Bayesian computation scheme for parameter inference and model selection in dynamical systems" - reference.

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.experimental.ode import odeint
import numpy as np
import time
from jax.lib import xla_bridge
from jax import random
#Look the necessary modules.
import numpy as np

#Read in the experimental data.
IL6_data_pS1 = np.loadtxt('IL6_data_pS1_mean_RPE1.txt')
IL6_data_pS3 = np.loadtxt('IL6_data_pS3_mean_RPE1.txt')
IL27_data_pS1 = np.loadtxt('IL27_data_pS1_mean_RPE1.txt')
IL27_data_pS3 = np.loadtxt('IL27_data_pS3_mean_RPE1.txt')
IL6_final = [IL6_data_pS1,IL6_data_pS3]
IL27_final = [IL27_data_pS1,IL27_data_pS3]

#Define the ordinary differential equation models.
#Hyper IL6 mathematical model where the terms involving the parameter beta apply only to hypothesis 1 and 
#the terms involving the parameter gamma apply only to hypothesis 2.
@jit
def dIL6_dt(R,t,par):
   k1ap, k1am, k3ap, k3am, qx, d1, d3, r10, s10, s30, r1p, r1m, r2p, r2m, beta, gamma = par   
   return jnp.array([
         -r1p*R[0]*R[1]+r1m*R[2]-beta*R[0]-gamma*(R[20]+R[21])*R[0],
         -r1p*R[0]*R[1]+r1m*R[2],
         r1p*R[0]*R[1]-r1m*R[2]-2*r2p*pow(R[2],2)+2*r2m*R[3]-beta*R[2]-gamma*(R[20]+R[21])*R[2],
         r2p*pow(R[2],2)-r2m*R[3]-2*k1ap*R[3]*R[4]+k1am*R[6]-2*k3ap*R[3]*R[5]+k3am*R[7]+k1am*R[8]+k3am*R[9]-beta*R[3]-gamma*(R[20]+R[21])*R[3],
         -k1ap*R[4]*(R[6]+R[7]+R[8]+R[9])+k1am*(2*R[10]+R[16]+R[11]+R[18])-2*k1ap*R[3]*R[4]+k1am*R[6]+d1*R[20],
         -k3ap*R[5]*(R[6]+R[7]+R[8]+R[9])+k3am*(2*R[13]+R[16]+R[14]+R[17])-2*k3ap*R[3]*R[5]+k3am*R[7]+d3*R[21],
         2*k1ap*R[3]*R[4]-k1am*R[6]-k1ap*R[6]*R[4]+2*k1am*R[10]-k3ap*R[6]*R[5]+k3am*R[16]-qx*R[6]+k1am*R[11]+k3am*R[18]-beta*R[6]-gamma*(R[20]+R[21])*R[6],
         2*k3ap*R[3]*R[5]-k3am*R[7]-k3ap*R[7]*R[5]+2*k3am*R[13]-k1ap*R[7]*R[4]+k1am*R[16]-qx*R[7]+k3am*R[14]+k1am*R[17]-beta*R[7]-gamma*(R[20]+R[21])*R[7],
         -k1ap*R[4]*R[8]+k1am*R[11]-k3ap*R[5]*R[8]+k3am*R[17]+qx*R[6]-k1am*R[8]+2*k1am*R[12]+k3am*R[19]-beta*R[8]-gamma*(R[20]+R[21])*R[8],
         -k3ap*R[5]*R[9]+k3am*R[14]-k1ap*R[4]*R[9]+k1am*R[18]+qx*R[7]-k3am*R[9]+2*k3am*R[15]+k1am*R[19]-beta*R[9]-gamma*(R[20]+R[21])*R[9],
         k1ap*R[4]*R[6]-2*k1am*R[10]-2*qx*R[10]-beta*R[10]-gamma*(R[20]+R[21])*R[10],
         k1ap*R[8]*R[4]-k1am*R[11]+2*qx*R[10]-(qx+k1am)*R[11]-beta*R[11]-gamma*(R[20]+R[21])*R[11],
         qx*R[11]-2*k1am*R[12]-beta*R[12]-gamma*(R[20]+R[21])*R[12],
         k3ap*R[5]*R[7]-2*k3am*R[13]-2*qx*R[13]-beta*R[13]-gamma*(R[20]+R[21])*R[13],
         k3ap*R[9]*R[5]-k3am*R[14]+2*qx*R[13]-(qx+k3am)*R[14]-beta*R[14]-gamma*(R[20]+R[21])*R[14],
         qx*R[14]-2*k3am*R[15]-beta*R[15]-gamma*(R[20]+R[21])*R[15], 
         k1ap*R[4]*R[7]-k1am*R[16]+k3ap*R[6]*R[5]-k3am*R[16]-(qx+qx)*R[16]-beta*R[16]-gamma*(R[20]+R[21])*R[16],
         qx*R[16]+k3ap*R[8]*R[5]-k3am*R[17]-qx*R[17]-k1am*R[17]-beta*R[17]-gamma*(R[20]+R[21])*R[17],
         qx*R[16]+k1ap*R[9]*R[4]-k1am*R[18]-qx*R[18]-k3am*R[18]-beta*R[18]-gamma*(R[20]+R[21])*R[18],
         qx*R[18]+qx*R[17]-k1am*R[19]-k3am*R[19]-beta*R[19]-gamma*(R[20]+R[21])*R[19],
         k1am*(R[8]+R[11]+R[17]+R[19])+2*k1am*R[12]-d1*R[20],
         k3am*(R[9]+R[14]+R[18]+R[19])+2*k3am*R[15]-d3*R[21],
         ])

v_dIL6_dt = vmap(dIL6_dt, in_axes=(0, None, 0))

@jit
def solve_dIL6_dt_batch(y0, t, params):
    solution = odeint(v_dIL6_dt, y0, t, params)
    return solution

#IL27 mathematical model where the terms involving the parameter beta apply only to hypothesis 1 and 
#the terms involving the parameter gamma apply only to hypothesis 2.

@jit
def dIL27_dt(R,t,par):
   k1ap, k1am, k1bp, k1bm, k3ap, k3am, k3bp, k3bm, qx, d1, d3, r10, r20, s10, s30, r1p, r1m, r2p, r2m, beta, gamma = par   
   return jnp.array([
         -r2p*R[0]*R[3]+r2m*R[4]-beta*R[0]-gamma*(R[31]+R[32])*R[0],
         -r1p*R[1]*R[2]+r1m*R[3]-beta*R[1]-gamma*(R[31]+R[32])*R[1],
         -r1p*R[1]*R[2]+r1m*R[3],
         r1p*R[1]*R[2]-r1m*R[3]-r2p*R[0]*R[3]+r2m*R[4]-beta*R[3]-gamma*(R[31]+R[32])*R[3],         
         r2p*R[0]*R[3]-r2m*R[4]-(k1ap+k1bp)*R[4]*R[5]+k1am*R[7]+k1bm*R[8]-(k3ap+k3bp)*R[4]*R[6]+k3am*R[9]+k3bm*R[10]+k1am*R[11]+k1bm*R[12]+k3am*R[13]+k3bm*R[14]-beta*R[4]-gamma*(R[31]+R[32])*R[4],
         -k1ap*R[5]*(R[4]+R[8]+R[12]+R[10]+R[14])+k1am*(R[7]+R[15]+R[17]+R[23]+R[27])-k1bp*R[5]*(R[4]+R[7]+R[11]+R[9]+R[13])+k1bm*(R[8]+R[15]+R[16]+R[24]+R[26])+d1*R[31],
         -k3ap*R[6]*(R[4]+R[8]+R[12]+R[10]+R[14])+k3am*(R[9]+R[24]+R[28]+R[19]+R[21])-k3bp*R[6]*(R[4]+R[7]+R[11]+R[9]+R[13])+k3bm*(R[10]+R[23]+R[25]+R[19]+R[20])+d3*R[32],
         k1ap*R[5]*R[4]-k1am*R[7]-k1bp*R[5]*R[7]+k1bm*R[15]-k3bp*R[6]*R[7]+k3bm*R[23]-qx*R[7]+k1bm*R[17]+k3bm*R[27]-beta*R[7]-gamma*(R[31]+R[32])*R[7],
         k1bp*R[5]*R[4]-k1bm*R[8]-k1ap*R[5]*R[8]+k1am*R[15]-k3ap*R[6]*R[8]+k3am*R[24]-qx*R[8]+k1am*R[16]+k3am*R[26]-beta*R[8]-gamma*(R[31]+R[32])*R[8],
         k3ap*R[6]*R[4]-k3am*R[9]-k3bp*R[6]*R[9]+k3bm*R[19]-k1bp*R[5]*R[9]+k1bm*R[24]-qx*R[9]+k3bm*R[21]+k1bm*R[28]-beta*R[9]-gamma*(R[31]+R[32])*R[9],
         k3bp*R[6]*R[4]-k3bm*R[10]-k3ap*R[6]*R[10]+k3am*R[19]-k1ap*R[5]*R[10]+k1am*R[23]-qx*R[10]+k3am*R[20]+k1am*R[25]-beta*R[10]-gamma*(R[31]+R[32])*R[10],
         -k1bp*R[11]*R[5]+k1bm*R[16]-k3bp*R[11]*R[6]+k3bm*R[25]+qx*R[7]-k1am*R[11]+k1bm*R[18]+k3bm*R[29]-beta*R[11]-gamma*(R[31]+R[32])*R[11],
         -k1ap*R[12]*R[5]+k1am*R[17]-k3ap*R[12]*R[6]+k3am*R[28]+qx*R[8]-k1bm*R[12]+k1am*R[18]+k3am*R[30]-beta*R[12]-gamma*(R[31]+R[32])*R[12],
         -k3bp*R[13]*R[6]+k3bm*R[20]-k1bp*R[13]*R[5]+k1bm*R[26]+qx*R[9]-k3am*R[13]+k3bm*R[22]+k1bm*R[30]-beta*R[13]-gamma*(R[31]+R[32])*R[13],
         -k3ap*R[14]*R[6]+k3am*R[21]-k1ap*R[14]*R[5]+k1am*R[27]+qx*R[10]-k3bm*R[14]+k3am*R[22]+k1am*R[29]-beta*R[14]-gamma*(R[31]+R[32])*R[14],
         k1ap*R[5]*R[8]-k1am*R[15]+k1bp*R[7]*R[5]-k1bm*R[15]-(qx+qx)*R[15]-beta*R[15]-gamma*(R[31]+R[32])*R[15],
         k1bp*R[11]*R[5]-k1bm*R[16]+qx*R[15]-qx*R[16]-k1am*R[16]-beta*R[16]-gamma*(R[31]+R[32])*R[16],
         k1ap*R[5]*R[12]-k1am*R[17]+qx*R[15]-qx*R[17]-k1bm*R[17]-beta*R[17]-gamma*(R[31]+R[32])*R[17],
         qx*R[17]+qx*R[16]-(k1am+k1bm)*R[18]-beta*R[18]-gamma*(R[31]+R[32])*R[18],
         k3ap*R[6]*R[10]-k3am*R[19]+k3bp*R[9]*R[6]-k3bm*R[19]-(qx+qx)*R[19]-beta*R[19]-gamma*(R[31]+R[32])*R[19],
         k3bp*R[13]*R[6]-k3bm*R[20]+qx*R[19]-qx*R[20]-k3am*R[20]-beta*R[20]-gamma*(R[31]+R[32])*R[20],
         k3ap*R[6]*R[14]-k3am*R[21]+qx*R[19]-qx*R[21]-k3bm*R[21]-beta*R[21]-gamma*(R[31]+R[32])*R[21],
         qx*R[21]+qx*R[20]-(k3am+k3bm)*R[22]-beta*R[22]-gamma*(R[31]+R[32])*R[22],
         k1ap*R[5]*R[10]-k1am*R[23]+k3bp*R[7]*R[6]-k3bm*R[23]-(qx+qx)*R[23]-beta*R[23]-gamma*(R[31]+R[32])*R[23],
         k3ap*R[6]*R[8]-k3am*R[24]+k1bp*R[9]*R[5]-k1bm*R[24]-(qx+qx)*R[24]-beta*R[24]-gamma*(R[31]+R[32])*R[24],
         k3bp*R[11]*R[6]-k3bm*R[25]+qx*R[23]-qx*R[25]-k1am*R[25]-beta*R[25]-gamma*(R[31]+R[32])*R[25],
         k1bp*R[13]*R[5]-k1bm*R[26]+qx*R[24]-qx*R[26]-k3am*R[26]-beta*R[26]-gamma*(R[31]+R[32])*R[26],
         k1ap*R[5]*R[14]-k1am*R[27]+qx*R[23]-qx*R[27]-k3bm*R[27]-beta*R[27]-gamma*(R[31]+R[32])*R[27],
         k3ap*R[6]*R[12]-k3am*R[28]+qx*R[24]-qx*R[28]-k1bm*R[28]-beta*R[28]-gamma*(R[31]+R[32])*R[28],
         qx*R[27]+qx*R[25]-(k1am+k3bm)*R[29]-beta*R[29]-gamma*(R[31]+R[32])*R[29],
         qx*R[28]+qx*R[26]-(k3am+k1bm)*R[30]-beta*R[30]-gamma*(R[31]+R[32])*R[30],
         k1am*(R[11]+R[16]+R[18]+R[25]+R[29])+k1bm*(R[12]+R[17]+R[18]+R[28]+R[30])-d1*R[31],
         k3am*(R[13]+R[26]+R[30]+R[20]+R[22])+k3bm*(R[14]+R[27]+R[29]+R[21]+R[22])-d3*R[32]
         ])

v_dIL27_dt = vmap(dIL27_dt, in_axes=(0, None, 0))

@jit
def solve_dIL27_dt_batch(y0, t, params):
    solution = odeint(v_dIL27_dt, y0, t, params, rtol=1.4e-3, atol=1.4e-3)
    return solution

#List of the possible models, 0 = hypotheis 1 and 1 = hypothesis 2.
models = [0,1]
model_probs = [0.5,0.5] #Prior probability of each model (equally likely).

#Function to run the first iteration of the hypothesis selection. A hypothesis is chosen by at random and the parameters for
#this hypothesis are sampled from their prior distributions. The distance is computed between the model and the data and stored in a vector.
#This process is repeated until there are N accepted parameters sets from hypothesis 1 and 2 combined.
# def first_iteration(N, epsilon):
#     print('Running iteration: 0')
#     batch_size = N

#     # Initialize JAX random key
#     key = random.PRNGKey(0)

#     # Function to run batched simulation
#     @jit
#     def batch_simulation(key):
#         # Generate random parameters for this batch using JAX random functions
#         k1ap = jnp.power(10, random.uniform(key, (batch_size,), minval=-7, maxval=1))
#         k1am = jnp.power(10, random.uniform(key, (batch_size,), minval=-2, maxval=1))
#         k1bp = jnp.power(10, random.uniform(key, (batch_size,), minval=-7, maxval=1))
#         k1bm = jnp.power(10, random.uniform(key, (batch_size,), minval=-2, maxval=1))
#         k3ap = jnp.power(10, random.uniform(key, (batch_size,), minval=-7, maxval=1))
#         k3am = jnp.power(10, random.uniform(key, (batch_size,), minval=-2, maxval=1))
#         k3bp = jnp.power(10, random.uniform(key, (batch_size,), minval=-7, maxval=1))
#         k3bm = jnp.power(10, random.uniform(key, (batch_size,), minval=-2, maxval=1))
#         qx = jnp.power(10, random.uniform(key, (batch_size,), minval=-3, maxval=2))
#         d1 = jnp.power(10, random.uniform(key, (batch_size,), minval=-5, maxval=-2))
#         d3 = jnp.power(10, random.uniform(key, (batch_size,), minval=-5, maxval=-2))
#         r10 = 12.7 + 6.35 * random.normal(key, (batch_size,))
#         r20 = 33.8 + 16.9 * random.normal(key, (batch_size,))
#         s10 = 300 + 100 * random.normal(key, (batch_size,))
#         s30 = 400 + 100 * random.normal(key, (batch_size,))
#         r2p = jnp.power(10, random.uniform(key, (batch_size,), minval=-2, maxval=3))
#         r2m = jnp.power(10, random.uniform(key, (batch_size,), minval=-3, maxval=1))
#         # Generate random parameters for HypIL-6 model
#         r1p_il6 = jnp.power(10, -3 + 1.5 * random.normal(key, (batch_size,)))
#         r1m_il6 = jnp.power(10, -3.9 + 1.96 * random.normal(key, (batch_size,)))
#         beta = jnp.where(random.choice(key,a=jnp.array([0, 1]), shape=(batch_size,), replace=True), jnp.power(10, random.uniform(key, (batch_size,), minval=-5, maxval=-1)), 0)

#         # Generate random parameters for IL-27 model
#         r1p_il27 = jnp.power(10, -2.34 + 1.17 * random.normal(key, (batch_size,)))
#         r1m_il27 = jnp.power(10, -2.82 + 1.41 * random.normal(key, (batch_size,)))
#         gamma = jnp.where(random.choice(key, a=jnp.array([0, 1]), shape=(batch_size,), replace=True), 0, jnp.power(10, random.uniform(key, (batch_size,), minval=-5, maxval=-1)))

#         # Combine parameters into a batched parameter set
#         params_batch_IL6 = jnp.stack((k1ap, k1am, k3ap, k3am, qx, d1, d3, r10, s10, s30, r1p_il6, r1m_il6, r2p, r2m, beta, gamma), axis=-1)
#         params_batch_IL27 = jnp.stack((k1ap, k1am, k1bp, k1bm, k3ap, k3am, k3bp, k3bm, qx, d1, d3, r10, r20, s10, s30, r1p_il27, r1m_il27, r2p, r2m, beta, gamma), axis=-1)

#         # Initial conditions
#         R0_IL6 = jnp.stack([r10,jnp.full(batch_size, 10),jnp.zeros(batch_size),jnp.zeros(batch_size),s10,s30,jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size)]).T #Initial concentrations.
#         R0_IL27 = jnp.stack([r10,r20,jnp.full(batch_size, 2),jnp.zeros(batch_size),jnp.zeros(batch_size),s10,s30,jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),jnp.zeros(batch_size),]).T #Initial concentrations.
        
#         #Set the timecourse for the integration (units of seconds).
#         tt = jnp.linspace(0.0, 108, 180)

#         # Run simulations
#         R_IL6 = solve_dIL6_dt_batch(R0_IL6, tt, params_batch_IL6)
#         print(R0_IL27.shape, tt.shape, params_batch_IL27.shape)
#         R_IL27 = solve_dIL27_dt_batch(R0_IL27, tt, params_batch_IL27)

#         # Extract relevant outputs for each model
#         pS1_IL6 = R_IL6[:, :, 8] + R_IL6[:, :, 11] + 2 * R_IL6[:, :, 12] + R_IL6[:, :, 17] + R_IL6[:, :, 19] + R_IL6[:, :, 20]
#         pS3_IL6 = R_IL6[:, :, 9] + R_IL6[:, :, 14] + 2 * R_IL6[:, :, 15] + R_IL6[:, :, 18] + R_IL6[:, :, 19] + R_IL6[:, :, 21]

#         pS1_IL27 = R_IL27[:, :, 11] + R_IL27[:, :, 12] + R_IL27[:, :, 16] + R_IL27[:, :, 17] + 2 * R_IL27[:, :, 18] + R_IL27[:, :, 25] + R_IL27[:, :, 28] + R_IL27[:, :, 29] + R_IL27[:, :, 30] + R_IL27[:, :, 31]
#         pS3_IL27 = R_IL27[:, :, 13] + R_IL27[:, :, 14] + R_IL27[:, :, 20] + R_IL27[:, :, 21] + 2 * R_IL27[:, :, 22] + R_IL27[:, :, 26] + R_IL27[:, :, 27] + R_IL27[:, :, 29] + R_IL27[:, :, 30] + R_IL27[:, :, 32]

#         # Time points for data
#         time_points = jnp.array([0, 5, 15, 30, 60, 90, 120, 180])

#         pS1_IL27_norm = pS1_IL27[time_points[4]]
#         pS3_IL27_norm = pS3_IL27[time_points[4]]

#         IL6_pS1_dis = jnp.sum(jnp.power(pS1_IL6[time_points] / pS1_IL27_norm - IL6_final[0][:,jnp.newaxis], 2), axis=0)
#         IL6_pS3_dis = jnp.sum(jnp.power(pS3_IL6[time_points] / pS3_IL27_norm - IL6_final[1][:,jnp.newaxis], 2), axis=0)
#         IL27_pS1_dis = jnp.sum(jnp.power(pS1_IL27[time_points] / pS1_IL27_norm - IL27_final[0][:,jnp.newaxis], 2), axis=0)
#         IL27_pS3_dis = jnp.sum(jnp.power(pS3_IL27[time_points] / pS3_IL27_norm - IL27_final[1][:,jnp.newaxis], 2), axis=0)

#         distances = jnp.sqrt(IL6_pS1_dis + IL6_pS3_dis + IL27_pS1_dis + IL27_pS3_dis)
#         return distances

#     # Execute the batched simulation
#     distances = batch_simulation(key)
#     print("completed batched simulation", distances)

#     # # Acceptance criteria (batched)
#     # accepted_models = []
#     # accepted_params_hyp1 = jnp.empty((0, 25))  # Adjust dimensions as per your parameter set
#     # accepted_params_hyp2 = jnp.empty((0, 25))  # Adjust dimensions as per your parameter set

#     # number = 0
#     # for i, (dist2, params_batch_IL6, params_batch_IL27) in enumerate(zip(distances, outputs_batch_IL6, outputs_batch_IL17)):
#     #     print(i)
#     #     if dist2 < epsilon:
#     #         model = i % 2  # Assign model based on even or odd index
#     #         accepted_models.append(model)
#     #         if model == 0:
#     #             results_hyp1 = jnp.hstack((results_hyp1, dist2))
#     #             accepted_params_hyp1 = jnp.vstack((accepted_params_hyp1, params_batch_IL6))
#     #         else:
#     #             results_hyp2 = jnp.hstack((results_hyp2, dist2))
#     #             accepted_params_hyp2 = jnp.vstack((accepted_params_hyp2, params_batch_IL27))
#     #         number += 1

#     # # Compute weights for accepted parameter sets
#     # weights_hyp1 = jnp.ones((len(accepted_params_hyp1), 1)) / len(accepted_params_hyp1)
#     # weights_hyp2 = jnp.ones((len(accepted_params_hyp2), 1)) / len(accepted_params_hyp2)

#     # # Print information about the first run of the ABC
#     # truns = N  # Example value, adjust as needed
#     # print('Acceptance rate for iteration 0: ' + str(N * 100 / truns))
#     # print('Epsilon = ' + str(epsilon))
#     # print('Total runs = ' + str(truns))

#     # # Return results (distances, accepted parameters, weights, total number of runs)
#     # results_hyp1_final = jnp.hstack((jnp.reshape(results_hyp1, (len(accepted_params_hyp1), 1)), accepted_params_hyp1, weights_hyp1))
#     # results_hyp2_final = jnp.hstack((jnp.reshape(results_hyp2, (len(accepted_params_hyp2), 1)), accepted_params_hyp2, weights_hyp2))

#     # # Stack accepted models for returning
#     # accepted_models_final = jnp.stack(accepted_models)

#     # # Example print outputs
#     # print("Results Hypothesis 1:")
#     # print(results_hyp1_final)
#     # print("Results Hypothesis 2:")
#     # print(results_hyp2_final)
#     # print("Accepted Models:")
#     # print(accepted_models_final)

#     # Example return statement (adjust as per your function structure)
#     # return [results_hyp1_final, results_hyp2_final, truns, accepted_models_final]

def first_iteration(N, epsilon):
    print('Running iteration: 0')
    batch_size = N

    # Initialize JAX random key
    key = random.PRNGKey(0)

    # Function to run batched simulation
    def generate_batch_simulation(key, batch_size):
        np.random.seed(0)
        k1ap = np.power(10, np.random.uniform(-7, 1, batch_size))
        k1am = np.power(10, np.random.uniform(-2, 1, batch_size))
        k1bp = np.power(10, np.random.uniform(-7, 1, batch_size))
        k1bm = np.power(10, np.random.uniform(-2, 1, batch_size))
        k3ap = np.power(10, np.random.uniform(-7, 1, batch_size))
        k3am = np.power(10, np.random.uniform(-2, 1, batch_size))
        k3bp = np.power(10, np.random.uniform(-7, 1, batch_size))
        k3bm = np.power(10, np.random.uniform(-2, 1, batch_size))
        qx = np.power(10, np.random.uniform(-3, 2, batch_size))
        d1 = np.power(10, np.random.uniform(-5, -2, batch_size))
        d3 = np.power(10, np.random.uniform(-5, -2, batch_size))
        r10 = 12.7 + 6.35 * np.random.normal(size=batch_size)
        r20 = 33.8 + 16.9 * np.random.normal(size=batch_size)
        s10 = 300 + 100 * np.random.normal(size=batch_size)
        s30 = 400 + 100 * np.random.normal(size=batch_size)
        r2p = np.power(10, np.random.uniform(-2, 3, batch_size))
        r2m = np.power(10, np.random.uniform(-3, 1, batch_size))
        r1p_il6 = np.power(10, -3 + 1.5 * np.random.normal(size=batch_size))
        r1m_il6 = np.power(10, -3.9 + 1.96 * np.random.normal(size=batch_size))
        beta = np.where(np.random.choice([0, 1], size=batch_size, replace=True), 
                        np.power(10, np.random.uniform(-5, -1, batch_size)), 0)
        r1p_il27 = np.power(10, -2.34 + 1.17 * np.random.normal(size=batch_size))
        r1m_il27 = np.power(10, -2.82 + 1.41 * np.random.normal(size=batch_size))
        gamma = np.where(np.random.choice([0, 1], size=batch_size, replace=True), 0, 
                         np.power(10, np.random.uniform(-5, -1, batch_size)))

        # Combine parameters into a batched parameter set
        params_batch_IL6 = np.stack((k1ap, k1am, k3ap, k3am, qx, d1, d3, r10, s10, s30, 
                                     r1p_il6, r1m_il6, r2p, r2m, beta, gamma), axis=-1)
        params_batch_IL27 = np.stack((k1ap, k1am, k1bp, k1bm, k3ap, k3am, k3bp, k3bm, qx, 
                                      d1, d3, r10, r20, s10, s30, r1p_il27, r1m_il27, r2p, 
                                      r2m, beta, gamma), axis=-1)

        # Initial conditions
        zeros = np.zeros(batch_size)
        R0_IL6 = np.stack([r10, np.full(batch_size, 10), zeros, zeros, s10, s30, zeros, 
                           zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, 
                           zeros, zeros, zeros, zeros, zeros, zeros, zeros], axis=-1)
        R0_IL27 = np.stack([r10, r20, np.full(batch_size, 2), zeros, zeros, s10, s30, zeros, 
                            zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, 
                            zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, 
                            zeros, zeros, zeros, zeros, zeros, zeros, zeros], axis=-1)

        # Convert to JAX arrays for ODE solver
        R0_IL6 = jnp.array(R0_IL6)
        R0_IL27 = jnp.array(R0_IL27)
        params_batch_IL6 = jnp.array(params_batch_IL6)
        params_batch_IL27 = jnp.array(params_batch_IL27)
        tt = jnp.linspace(0.0, 10800, 18001) # jnp.linspace(0.0, 108, 181)

        return R0_IL6, R0_IL27, params_batch_IL6, params_batch_IL27, tt
    
    def compute_distances(R_IL6, R_IL27):

        # Convert results back to NumPy for post-processing
        R_IL6 = np.array(R_IL6)
        R_IL27 = np.array(R_IL27)

        # Extract relevant outputs for each model
        pS1_IL6 = R_IL6[:, :, 8] + R_IL6[:, :, 11] + 2 * R_IL6[:, :, 12] + R_IL6[:, :, 17] + R_IL6[:, :, 19] + R_IL6[:, :, 20]
        pS3_IL6 = R_IL6[:, :, 9] + R_IL6[:, :, 14] + 2 * R_IL6[:, :, 15] + R_IL6[:, :, 18] + R_IL6[:, :, 19] + R_IL6[:, :, 21]

        pS1_IL27 = R_IL27[:, :, 11] + R_IL27[:, :, 12] + R_IL27[:, :, 16] + R_IL27[:, :, 17] + 2 * R_IL27[:, :, 18] + R_IL27[:, :, 25] + R_IL27[:, :, 28] + R_IL27[:, :, 29] + R_IL27[:, :, 30] + R_IL27[:, :, 31]
        pS3_IL27 = R_IL27[:, :, 13] + R_IL27[:, :, 14] + R_IL27[:, :, 20] + R_IL27[:, :, 21] + 2 * R_IL27[:, :, 22] + R_IL27[:, :, 26] + R_IL27[:, :, 27] + R_IL27[:, :, 29] + R_IL27[:, :, 30] + R_IL27[:, :, 32]

        # Time points for data
        time_points = np.array([0, 5, 15, 30, 60, 90, 120, 180])

        pS1_IL27_norm = pS1_IL27[time_points[4]]
        pS3_IL27_norm = pS3_IL27[time_points[4]]

        IL6_pS1_dis = np.sum(np.power(pS1_IL6[time_points] / pS1_IL27_norm - IL6_final[0][:, np.newaxis], 2), axis=0)
        IL6_pS3_dis = np.sum(np.power(pS3_IL6[time_points] / pS3_IL27_norm - IL6_final[1][:, np.newaxis], 2), axis=0)
        IL27_pS1_dis = np.sum(np.power(pS1_IL27[time_points] / pS1_IL27_norm - IL27_final[0][:, np.newaxis], 2), axis=0)
        IL27_pS3_dis = np.sum(np.power(pS3_IL27[time_points] / pS3_IL27_norm - IL27_final[1][:, np.newaxis], 2), axis=0)

        distances = np.sqrt(IL6_pS1_dis + IL6_pS3_dis + IL27_pS1_dis + IL27_pS3_dis)
        return distances

    # Execute the batched simulation
    start = time.time()
    R0_IL6, R0_IL27, params_batch_IL6, params_batch_IL27, tt = generate_batch_simulation(key, batch_size)
    print(f"time required for param generation={time.time()- start}")
    start = time.time()
    # Run simulations
    R_IL6 = solve_dIL6_dt_batch(R0_IL6, tt, params_batch_IL6)
    print(f"time required for IL6 simulation={time.time()- start}, {R_IL6}")
    # start = time.time()
    # R_IL27 = solve_dIL27_dt_batch(R0_IL27, tt, params_batch_IL27)
    # print(f"time required for IL27 simulation={time.time()- start}")
    # start = time.time()
    # distances = compute_distances(R_IL6, R_IL27)
    # print(f"time required for distance measurement={time.time()- start}")
    # print("Completed batched simulation", distances)

# start = time.time()
# first_iteration(1, 10)  # Adjust parameters as needed
# print(f"time required for dry run={time.time()- start}")

# Example usage
start = time.time()
first_iteration(100,100)  # Adjust parameters as needed
print(f"time required={time.time()- start}")
# #Function for the other iterations of the ABC-SMC model selection algorithm where the parameters are sampled from the previous posteriors.
# def other_iterations(N,it):
#    print('Running iteration: ' + str(it+1))

#    epsilon = epsilons[it]
#    len1 = len(ABC_hypothesis1[it])
#    len2 = len(ABC_hypothesis2[it])
#    lens = [len1,len2]
#    p_list_hyp1 = [i for i in range(len1)]
#    p_list_hyp2 = [i for i in range(len2)]
#    p_lists = [p_list_hyp1,p_list_hyp2]

#    #Upper and lower bounds for the uniform distributions of the priors for each parameter.
#    lower_bounds = [-10,-15,-2,-3,-5,-10,-15,-2,-3,-5,-7,-2,-7,-2,-7,-2,-7,-2,-3,-5,-5, 0,  0,  0,   0]
#    upper_bounds = [  5,  5, 3, 1,-1,  5,  5, 3, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 2,-2,-2,40,100,800,1000] 
   
#    #Compute uniform areas to sample within in order to perturb the parameters.
#    both_hyps = [ABC_hypothesis1[it],ABC_hypothesis2[it]]
#    sigmas = []
#    for hyp in range(2):
#       ranges = []
#       for i in range(25):
#          if i in [21,22,23,24]:
#             r1 = np.max(both_hyps[hyp][:,i+1]) - np.min(both_hyps[hyp][:,i+1])
#          else:
#             r1 = np.max(np.log10(both_hyps[hyp][:,i+1])) - np.min(np.log10(both_hyps[hyp][:,i+1]))
#          ranges.append(r1)
#       ranges_arr = np.asarray(ranges)
#       sigma = 0.2*ranges_arr
#       sigmas.append(sigma)

#    #Empty list for the accepted model selected.
#    accepted_models = []
#    #Empty arrays for the prior samples.
#    priors_hyp1 = np.empty((0,25))
#    priors_hyp2 = np.empty((0,25))
#    #Empty arrays for the accepted parameters of each hypothesis.
#    accepted_params_hyp1 = np.empty((0,25))
#    accepted_params_hyp2 = np.empty((0,25))
#    #Store the distance measures from the ABC.
#    results_hyp1 = np.empty((0))
#    results_hyp2 = np.empty((0))
#    #Store the weights for each hypothesis.
#    weights_hyp1 = np.empty((0))
#    weights_hyp2 = np.empty((0))

#    number = 0 #Counter for the population. 
#    truns = 0 #Count total number of runs in order to acceptance percentage.
#    while number < N: #Run the loop until N parameter sets are accepted.
#       model = np.random.choice(models, p=model_probs) #Select a hypothesis at random.
#       truns+=1
#       #Loop to sample parameters from the posterior distributions of the previous iteration. The parameters are then perturbed,
#       #using a uniform perturbation kernel, and if the new parameters all lie within their uniform priors, then they are used in the model integration,
#       #otherwise they are resampled.      
#       check = 0
#       while check < 1:
#          choice = np.random.choice(p_lists[model],1,p=both_hyps[model][:,26]) #Choose a random parameter set from previous iterations posteriors.
#          prior_sample = both_hyps[model][:,range(1,26)][choice]
#          #Generate new parameters through perturbation.
#          parameters = []
#          for i in range(25):
#             if i in [21,22,23,24]:
#                lower = prior_sample[0,i]-sigmas[model][i]
#                upper = prior_sample[0,i]+sigmas[model][i]
#             else:
#                lower = np.log10(prior_sample[0,i])-sigmas[model][i]
#                upper = np.log10(prior_sample[0,i])+sigmas[model][i]
#             parameter = np.random.uniform(lower, upper)
#             if i in [21,22,23,24]:
#                parameters.append(parameter)
#             else:
#                parameters.append(pow(10,parameter))
#          #Check that the new parameters are feasible given the priors.
#          check_out = 0
#          for ik in range(25):
#             if ik in [21,22,23,24]:
#                if parameters[ik] < lower_bounds[ik] or parameters[ik] > upper_bounds[ik]:
#                   check_out = 1
#             else:
#                if parameters[ik] < pow(10,lower_bounds[ik]) or parameters[ik] > pow(10,upper_bounds[ik]):
#                   check_out = 1
#          if check_out == 0:
#             check+=1

#       k1ap = parameters[10]
#       k1am = parameters[11]
#       k1bp = parameters[12]
#       k1bm = parameters[13]
#       k3ap = parameters[14]
#       k3am = parameters[15]
#       k3bp = parameters[16]
#       k3bm = parameters[17]
#       qx  = parameters[18]
#       d1   = parameters[19]
#       d3   = parameters[20]
#       r10 = parameters[21]
#       r20 = parameters[22]
#       s10   = parameters[23]
#       s30   = parameters[24]      
#       outputs = [] #Empty list for the model outputs.
#       paramsa = [] #Empty list for the parameters.
#       for stim in range(2): #Loop for the two cytokine models.
#          #Run the HypIL-6 model.
#          if stim == 0:
#             r1p = parameters[0]
#             r1m = parameters[1]
#             r2p = parameters[2]
#             r2m = parameters[3]
#             if model == 0: #Hypothesis 1.
#                beta = parameters[4]            
#                gamma = 0
#             else: #Hypothesis 2.
#                gamma = parameters[4]            
#                beta = 0               
#             if model == 0:
#                paramsa.append(np.array((r1p,r1m,r2p,r2m,beta)))
#             else:
#                paramsa.append(np.array((r1p,r1m,r2p,r2m,gamma)))
               
#             parset = np.array((k1ap, k1am, k3ap, k3am, qx, d1, d3, r10, s10, s30, r1p, r1m, r2p, r2m, beta, gamma)) #Parameter set to feed into the integrator.
#             R0 = np.array([r10,10,0,0,s10,s30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) #Initial concentrations.
#             R = integrate.odeint(dIL6_dt, R0, tt, args=(parset,)) #Integrate the mathematical model.
#             n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, n21, n22 = R.T #Model outputs for each variable.
#             pS1 = n9 + n12 + 2*n13 + n18 + n20 + n21 #Sum of variables containing pSTAT1.
#             pS3 = n10 + n15 + 2*n16 + n19 + n20 + n22 #Sum of vairables containing pSTAT3.                              
            
#          else:
#             r1p = parameters[5]
#             r1m = parameters[6]
#             r2p = parameters[7]
#             r2m = parameters[8]
#             if model == 0: #Hypothesis 1.
#                beta = parameters[9]            
#                gamma = 0
#             else: #Hypothesis 2.
#                gamma = parameters[9]            
#                beta = 0            
#             if model == 0:
#                paramsa.append(np.array((r1p,r1m,r2p,r2m,beta)))
#             else:
#                paramsa.append(np.array((r1p,r1m,r2p,r2m,gamma)))
               
#             parset = np.array((k1ap, k1am, k1bp, k1bm, k3ap, k3am, k3bp, k3bm, qx, d1, d3, r10, r20, s10, s30, r1p, r1m, r2p, r2m, beta, gamma)) #Parameter set to feed into the integrator.
#             R0 = np.array([r10,r20,2,0,0,s10,s30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) #Initial concentrations.
#             R = integrate.odeint(dIL27_dt, R0, tt, args=(parset,)) #Integrate the mathematical model.
#             n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, n21, n22, n23, n24, n25, n26, n27, n28, n29, n30, n31, n32, n33 = R.T #Model outputs for each vairable.
#             pS1 = n12 + n13 + n17 + n18 + 2*n19 + n26 + n29 + n30 + n31 + n32 #Sum of variables containing pSTAT1.
#             pS3 = n14 + n15 + n21 + n22 + 2*n23 + n27 + n28 + n30 + n31 + n33 #Sum of variables containing pSTAT3.
#          #Add model simulations for total pSTAT1 and pSTAT3 to outputs list.
#          outputs.append(np.array((pS1[0::100],pS3[0::100]))) #Append model outputs (cut to length 181) to the list.
         
#       paramsa.append(np.array((k1ap, k1am, k1bp, k1bm, k3ap, k3am, k3bp, k3bm, qx, d1, d3, r10, r20, s10, s30)))
#       paramsa = np.hstack((paramsa)) #Create an array of the sampled parameter values.   

#       #Take the values of the model simulations that correlate to the data time points.
#       time_points = np.array((0,5,15,30,60,90,120,180)) #Data time points.
#       IL6_pS1_model = np.empty((8,))
#       IL6_pS3_model = np.empty((8,))
#       IL27_pS1_model = np.empty((8,))
#       IL27_pS3_model = np.empty((8,))
#       for i in range(8):
#          IL6_pS1_model[i] = outputs[0][0,:][time_points[i]]
#          IL6_pS3_model[i] = outputs[0][1,:][time_points[i]]
#          IL27_pS1_model[i] = outputs[1][0,:][time_points[i]]
#          IL27_pS3_model[i] = outputs[1][1,:][time_points[i]]

#       #Normalise the model in the same way as the data by dividing by IL-27 time 30 minutes.
#       norm_point_s1 = IL27_pS1_model[3]
#       norm_point_s3 = IL27_pS3_model[3]
#       IL6_pS1_model_norm = IL6_pS1_model/norm_point_s1
#       IL6_pS3_model_norm = IL6_pS3_model/norm_point_s3
#       IL27_pS1_model_norm = IL27_pS1_model/norm_point_s1
#       IL27_pS3_model_norm = IL27_pS3_model/norm_point_s3

#       #Compute the distance between the model and the data.
#       IL6_pS1_dis = []
#       IL6_pS3_dis = []
#       IL27_pS1_dis = []
#       IL27_pS3_dis = []
#       for t in range(8):
#          IL6_pS1_dis.append(pow((IL6_pS1_model_norm[t] - IL6_final[0][t]),2))         
#          IL6_pS3_dis.append(pow((IL6_pS3_model_norm[t] - IL6_final[1][t]),2))
#          IL27_pS1_dis.append(pow((IL27_pS1_model_norm[t] - IL27_final[0][t]),2))
#          IL27_pS3_dis.append(pow((IL27_pS3_model_norm[t] - IL27_final[1][t]),2))

#       dist = np.sum((np.array((np.sum(IL6_pS1_dis),np.sum(IL6_pS3_dis),np.sum(IL27_pS1_dis),np.sum(IL27_pS3_dis)))))
#       dist2 = np.sqrt(dist)
      
#       #If the distance is less than epsilon, append distance and parameters to their respective lists and compute the weight
#       #for the accepted parameter set. Keep a list of the model which was accepted.
#       if dist2 < epsilon:
#          number+=1     
#          accepted_models.append(model)
#          denom_arr = []
#          for j in range(lens[model]):
#             weight = both_hyps[model][j,26]
#             params_row = both_hyps[model][j,1:26]
#             boxs_up = []
#             boxs_low = []
#             for i in range(25):
#                if i in [21,22,23,24]:
#                   boxs_up.append(params_row[i] + sigmas[model][i])
#                   boxs_low.append(params_row[i] - sigmas[model][i])
#                else:
#                   boxs_up.append(np.log10(params_row[i]) + sigmas[model][i])
#                   boxs_low.append(np.log10(params_row[i]) - sigmas[model][i])
#             outside = 0
#             for i in range(25):
#                if i in [21,22,23,24]:
#                   if parameters[i] < boxs_low[i] or parameters[i] > boxs_up[i]:
#                      outside = 1
#                else:
#                   if np.log10(parameters[i]) < boxs_low[i] or np.log10(parameters[i]) > boxs_up[i]:
#                      outside = 1                  
#             if outside == 1:
#                denom_arr.append(0)
#             else:
#                denom_arr.append(weight*np.prod(1/(2*sigmas[model])))
#          weight_param = 1/np.sum(denom_arr)
         
#          if model == 0:
#             weights_hyp1 = np.hstack((weights_hyp1,weight_param))
#             results_hyp1 = np.hstack((results_hyp1, dist2))
#             accepted_params_hyp1 = np.vstack((accepted_params_hyp1, paramsa))
#             priors_hyp1 = np.vstack((priors_hyp1, prior_sample))
#          else:
#             weights_hyp2 = np.hstack((weights_hyp2,weight_param))
#             results_hyp2 = np.hstack((results_hyp2, dist2))
#             accepted_params_hyp2 = np.vstack((accepted_params_hyp2, paramsa))
#             priors_hyp2 = np.vstack((priors_hyp2, prior_sample))            

#    #Normalise the weights
#    weights_hyp1_2 = weights_hyp1/np.sum(weights_hyp1)
#    weights_hyp1_3 = np.reshape(weights_hyp1_2, (len(weights_hyp1_2),1))
   
#    weights_hyp2_2 = weights_hyp2/np.sum(weights_hyp2)
#    weights_hyp2_3 = np.reshape(weights_hyp2_2, (len(weights_hyp2_2),1))   

#    #Print information about the first run of the ABC.
#    print('Acceptance rate for iteration ' + str(it+1) + ': ' + str(N*100/truns))
#    print('Epsilon = ' + str(epsilon))
#    print('Total runs = ' + str(truns))
#    #Return the results (distances, accepted parameters, weights, total number of runs).   
#    return [np.hstack((np.reshape(results_hyp1,(len(accepted_params_hyp1),1)),accepted_params_hyp1,weights_hyp1_3)),
#            np.hstack((np.reshape(results_hyp2,(len(accepted_params_hyp2),1)),accepted_params_hyp2,weights_hyp2_3)),truns,np.stack((accepted_models))]

# #Sample size for the ABC-SMC model selection.
# N = 10000
# #Set the array of epsilon values to be used.
# epsilons = [10,5,3,2.5,2.25,2,1.75,1.5,1.25,1.1,1,0.9,0.8,0.7]
# #Run the first iteration with a sufficiently large value of epsilon.
# first = first_iteration(N,100)
# model_select = [] #List of the accepted models.
# ABC_hypothesis1 = []
# ABC_hypothesis2 = []
# ABC_hypothesis1.append(first[0])
# ABC_hypothesis2.append(first[1])
# model_select.append(first[3])
# np.savetxt('Accepted_models_iteration_0.txt', model_select[0])
# #Run the sucessive iterations of the ABC.
# for itt in range(len(epsilons)):
#    run = other_iterations(N,itt)
#    ABC_hypothesis1.append(run[0])
#    ABC_hypothesis2.append(run[1])
#    model_select.append(run[3])
#    np.savetxt('Accepted_models_iteration_' + str(itt+1) + '.txt', model_select[itt+1])
#    if len(run[0]) == 0 or len(run[1]) == 0:
#       print(' ')
#       if len(run[0]) == 0:
#          print('Model selection completed - ran of out hypothesis 1 particles')
#       else:
#          print('Model selection completed - ran of out hypothesis 2 particles')
#       break

