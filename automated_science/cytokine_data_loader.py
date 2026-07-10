import numpy as np
import os

def load_and_serialize_cytokine():
    print("Loading empirical Cytokine T-Helper 1 datasets...")
    base_dir = "/Users/sourabh/Documents/Projects/workspace/thesis_project/jax-cytokine"
    
    # Read the data arrays
    IL6_pS1 = np.loadtxt(os.path.join(base_dir, "IL6_data_pS1_mean_TH1.txt"))
    IL6_pS3 = np.loadtxt(os.path.join(base_dir, "IL6_data_pS3_mean_TH1.txt"))
    
    # Create the serialized target format: (Timepoints, 2 distinct STAT trajectories)
    # The arrays are size 8 representing time points: (0, 5, 15, 30, 60, 90, 120, 180)
    time_points = np.array((0, 5, 15, 30, 60, 90, 120, 180))
    
    # We will serialize just the IL6 ground truth targets as Shape (8, 2)
    IL6_ground_truth = np.column_stack((IL6_pS1, IL6_pS3))
    
    out_dir = "/Users/sourabh/Documents/Projects/workspace/automated_science/data"
    np.save(os.path.join(out_dir, "hyper_IL6_time_points.npy"), time_points)
    np.save(os.path.join(out_dir, "hyper_IL6_ground_truth.npy"), IL6_ground_truth)
    
    # We also need the RPE1 Posteriors just in case baseline priors are requested
    rpe1_posteriors = np.loadtxt(os.path.join(base_dir, "RPE1_posteriors.txt"))
    # Save the minimums and maximums across the 10,000 posterior particles as parameter bounds!
    np.save(os.path.join(out_dir, "RPE1_bounds.npy"), np.column_stack((np.min(rpe1_posteriors, axis=0), np.max(rpe1_posteriors, axis=0))))
    print("Serialization Complete! Target tensor mapped.")

if __name__ == "__main__":
    load_and_serialize_cytokine()
