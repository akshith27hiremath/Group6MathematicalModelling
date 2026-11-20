# runs all the scripts, figures go into ../figures
import os
import sys

os.makedirs("../figures", exist_ok=True)

for name in ["eigen_analysis.py", "simulations.py", "phase_plane.py", "type1_diabetes.py"]:
    print("=====", name, "=====")
    os.system(sys.executable + " " + name)
