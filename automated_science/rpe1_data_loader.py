import numpy as np
import os

def serialize_rpe1_data():
    print("Serializing RPE1 Cytokine datasets...")
    base_dir = "/Users/sourabh/Documents/Projects/workspace/thesis_project/jax-cytokine"
    out_dir = "/Users/sourabh/Documents/Projects/workspace/automated_science/data"
    
    # IL6 RPE1 Data
    il6_pS1 = np.loadtxt(os.path.join(base_dir, "IL6_data_pS1_mean_RPE1.txt"))
    il6_pS3 = np.loadtxt(os.path.join(base_dir, "IL6_data_pS3_mean_RPE1.txt"))
    il6_ground_truth = np.column_stack((il6_pS1, il6_pS3))
    
    # IL27 RPE1 Data (For future Phase 8 expansion)
    il27_pS1 = np.loadtxt(os.path.join(base_dir, "IL27_data_pS1_mean_RPE1.txt"))
    il27_pS3 = np.loadtxt(os.path.join(base_dir, "IL27_data_pS3_mean_RPE1.txt"))
    il27_ground_truth = np.column_stack((il27_pS1, il27_pS3))
    
    time_points = np.array((0, 5, 15, 30, 60, 90, 120, 180))
    
    np.save(os.path.join(out_dir, "rpe1_il6_ground_truth.npy"), il6_ground_truth)
    np.save(os.path.join(out_dir, "rpe1_il27_ground_truth.npy"), il27_ground_truth)
    np.save(os.path.join(out_dir, "rpe1_time_points.npy"), time_points)
    
    print("RPE1 Data Serialization Complete.")

if __name__ == "__main__":
    serialize_rpe1_data()
