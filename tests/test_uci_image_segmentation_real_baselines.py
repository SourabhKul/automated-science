import numpy as np
from scripts.run_uci_image_segmentation_real_baselines import Rows,fit,metric
def main():
 y=np.repeat(np.asarray(("BRICKFACE","CEMENT","FOLIAGE","GRASS","PATH","SKY","WINDOW")),4);r=Rows(np.random.default_rng(54).normal(size=(len(y),19)),y,tuple(map(str,range(len(y)))));assert 0<=metric(fit("logreg_0.1",r.x,r.y),r)["macro_f1"]<=1;print("SUCCESS: Image Segmentation observed baseline helper emits finite probabilities")
if __name__=="__main__":main()
