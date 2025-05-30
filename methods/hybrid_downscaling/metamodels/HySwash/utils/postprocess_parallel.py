import os
import os.path as op
from bluemath_tk.core.io import load_model
import sys

case_num = int(sys.argv[1]) if len(sys.argv) > 1 else 0

root_dir="/home/grupos/geocean/valvanuz/HySwash/BlueMath/methods/hybrid_downscaling/metamodels/HySwash"
#output_dir = "/lustre/geocean/DATA/hidronas1/valva/Veggy_topo_alba"
output_dir = os.path.join(root_dir, "output_Veggy_Hs_mono")
templates_dir = os.path.join(root_dir, "templates", "VeggyBig")
export_dir = op.join(root_dir, "exported_Veggy_Hs_mono")



swash_model = load_model(op.join(export_dir, "swash_model.pkl"))
swash_model.load_cases()
#swash_model.build_cases([case_num])
#swash_model.run_cases([case_num])
swash_model.postprocess_cases([case_num])
