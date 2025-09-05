import sys
import os
import os.path as op
import numpy as np
from bluemath_tk.core.io import load_model

case_num=sys.argv[1]

root_dir = "/home/grupos/geocean/valvanuz/HySwash/BlueMath/methods/hybrid_downscaling/metamodels/HySwash/"
#output_dir = "/discos/rapido/outputVeggy"
output_dir = "/lustre/geocean/DATA/hidronas1/valva/Veggy_topo_alba"
templates_dir = op.join(root_dir, "templates", "VeggyBig")
export_dir = op.join(root_dir, "HyVeggy_exported")


mda = load_model(model_path=op.join(export_dir, "mda_model.pkl"))
depth_array=np.loadtxt(op.join(templates_dir, "depth.bot"))
swash_model=load_model(op.join(export_dir, "swash_model.pkl"))

os.environ["OMP_NUM_THREADS"] = "1"  

# case_num in format 0000
case_num_s = str(case_num).zfill(4)

swash_model.build_cases(mode="one_by_one",cases_to_build=[int(case_num)])
swash_model.run_cases(cases_to_run=[int(case_num)],launcher="/software/geocean/bin/launchSwash.sh")
swash_model.postprocess_cases(cases_to_postprocess=[int(case_num)],force=True)
