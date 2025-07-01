#!/bin/python3
import numpy as np
import pathlib
import json
import h5py
import tempfile
import subprocess
from itertools import product
import shutil
    
def __assemble_realizations(input_files, output_files, out_f=None):
    # out_path = "/home/sayan/pyddc/results9/"
    if out_f is None:
        out_path = pathlib.Path.cwd() / "results/"
    else:
        out_path = pathlib.Path.cwd() / out_f 

    if out_path.exists() and out_path.is_dir():
        shutil.rmtree(out_path)
    out_path.mkdir()

    for o, inp_lst in zip(output_files, input_files):
        o = str(out_path) + "/" + o.name + ".h5"
        fd = h5py.File(o, 'w')
        for i, inp_f in enumerate(inp_lst):
            try:
                inp = h5py.File(inp_f, 'r') 
                gr = list(inp.keys())[0]
                inp_gr = inp.get(gr)                    
                inp_gr.copy(source = inp_gr, dest = fd, name=gr + str(i))
                inp.close()
            except:
                continue
        print("Realizations copied to {}".format(o))
        fd.close() 

def assemble(out_f):
    # path = "/home/sayan/pyddc/outputs"
    p = pathlib.Path.cwd() / "outputs"
    # p = pathlib.Path(out_path)
    x_names = [i for i in p.iterdir() if not i.name.endswith('.h5')]
    x_outputs = [i.resolve() for i in p.iterdir() if i.name.endswith('.h5')]
    collect = []
    for i in x_names:
        filtered_outputs = []
        for j in x_outputs:
            if i.stem in j.stem: 
                filtered_outputs.append(j.resolve())
        collect.append(filtered_outputs)

    __assemble_realizations(collect, x_names, out_f)

def GenerateIOFiles(inp_jf_path):
    with open(inp_jf_path, "r") as jf:
        inputs = json.load(jf)

    multi_params = []
    param_vals = []
    for k, v in inputs.items():
        if isinstance(v, list):
            multi_params.append(k); param_vals.append(v)

    vals = list(product(*param_vals))

    param = [ele for ele in [multi_params] for _ in range(len(vals))]
    combinations = list(zip(param, vals)) 
    pv = []
    for key, val in combinations:
        pv.append(dict(zip(key, val)))

    parent = pathlib.Path.cwd()
    i_dir = parent / "inputs"; o_dir = parent / "outputs"; log_dir = parent / "logs"
    if i_dir.exists() and i_dir.is_dir():
        shutil.rmtree(i_dir)
    if o_dir.exists() and o_dir.is_dir():
        shutil.rmtree(o_dir)
    if log_dir.exists() and log_dir.is_dir():
        shutil.rmtree(log_dir)

    i_dir.mkdir(); o_dir.mkdir(); log_dir.mkdir()
    io = []
    for pid, items in enumerate(pv):
        inpf = tempfile.NamedTemporaryFile(suffix= ".json", dir=i_dir, mode="w+", delete=False)
        inputs.update(items)
        json.dump(inputs, inpf)
        outf = tempfile.NamedTemporaryFile(dir=o_dir, delete=False)
        io.append([inpf.name, outf.name, str(items), str(pid)]) 
    return io

def execute(inp_jf_path, out_folder, realizations=1):
    '''
    Execute the simulation jobs based on the input JSON file and assemble the results in the path specified by assemble_path.
    Args:
        inp_jf_path (str): Full path to the input JSON file containing simulation parameters.
        assemble_path (str): Path where the assembled results will be stored.
        realizations (int): Number of realizations to run for each set of parameters.
    Returns:
        None
    '''
    parent = pathlib.Path.cwd()
    io_files = GenerateIOFiles(inp_jf_path)
    dispatcher = [['python3', parent/'sc/simulation.py', i, o, data, pid, str(r)] for i, o, data, pid in io_files for r in range(1, realizations+1)]
    proc = []

    try:
        for cmd in dispatcher:
            proc.append(subprocess.Popen(cmd))
        for p in proc:
            p.wait()
    except KeyboardInterrupt:
        for p in proc:
            p.terminate()
    
    assemble(out_folder)
    print("jobs completed and realizations assembled in the output dir inside parent dir- ./pyddc.")

    i_dir = parent / "inputs"; o_dir = parent / "outputs"
    if i_dir.exists() and i_dir.is_dir():
        shutil.rmtree(i_dir)
    if o_dir.exists() and o_dir.is_dir():
        shutil.rmtree(o_dir)

if __name__ == "__main__":
    execute("/home/sayan/ctrans_mod/inputs.json", "results", realizations=1)
#     inp_jf_path = "/home/sayan/pyddc/inputs.json" # full path of input file
#     io_files = GenerateIOFiles(inp_jf_path)
#     realizations = 1
#     dispatcher = [['python3', '/home/sayan/pyddc/sc/simulation.py', i, o, data, pid, str(r)] for i, o, data, pid in io_files for r in range(1, realizations+1)]
#     proc = []

#     try:
#         for cmd in dispatcher:
#             proc.append(subprocess.Popen(cmd))
#         for p in proc:
#             p.wait()
#     except KeyboardInterrupt:
#         for p in proc:
#             p.terminate()
    
#     assemble()
#     print("jobs completed and realizations assembled in the output dir inside parent dir- ./pyddc.")
