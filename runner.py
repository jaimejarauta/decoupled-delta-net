"""Sandbox entry: python -I runner.py PROG.py INIT.npy|none TIME_LIMIT OUT.npy"""
import importlib.util, sys
import numpy as np
prog, init, tl, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
spec = importlib.util.spec_from_file_location("prog", prog)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
best = None if init == "none" else np.load(init)
np.save(out, np.asarray(m.solve(tl, best), dtype=np.float64))
