import numpy as np
import os

def generate_covid_data():
    # Extracted from thesis_project/Journal Paper ABC-SMC/ABC_IPU_GPU/model_with_plotting.py
    # Shape: (3, 49) -> Active, Recovered, Deaths
    country_data_train = np.array([
        [   155,    229,    322,    453,    655,    888,   1128,   1694,   2036,
           2502,   3089,   3858,   4636,   5883,   7375,   9172,  10149,  12462,
          15113,  17660,  21157,  24747,  27980,  31506,  35713,  41035,  47021,
          53578,  59138,  63927,  69176,  74386,  80589,  86498,  92472,  97689,
         101739, 105792, 110574, 115242, 119827, 124632, 128948, 132547, 135586,
         139422, 143626, 147577, 152271],
        [     2,      1,      1,      3,     45,     46,     46,     83,    149,
            160,    276,    414,    523,    589,    622,    724,    724,   1045,
           1045,   1439,   1966,   2335,   2749,   2941,   4025,   4440,   4440,
           6072,   7024,   7024,   8326,   9362,  10361,  10950,  12384,  13030,
          14620,  15729,  16847,  18278,  19758,  20996,  21815,  22837,  24392,
          26491,  28470,  30455,  32534],
        [     3,      7,     10,     12,     17,     21,     29,     34,     52,
             79,    107,    148,    197,    233,    366,    463,    631,    827,
           1016,   1266,   1441,   1809,   2158,   2503,   2978,   3405,   4032,
           4825,   5476,   6077,   6820,   7503,   8215,   9134,  10023,  10779,
          11591,  12428,  13155,  13915,  14681,  15362,  15887,  16523,  17127,
          17669,  18279,  18849,  19468]
    ], dtype=np.float32)

    # Transpose to shape (Time, Features) -> (49, 3) 
    # to match the generic BaseModel format (time_steps, dim)
    obs_data = country_data_train.T
    time_points = np.arange(49, dtype=np.float32)

    os.makedirs("automated_science/data", exist_ok=True)
    np.save("automated_science/data/covid_ground_truth.npy", obs_data)
    np.save("automated_science/data/covid_time_points.npy", time_points)
    print(f"Generated COVID ground truth data with shape {obs_data.shape}")

if __name__ == "__main__":
    generate_covid_data()
