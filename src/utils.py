"""
Auxiliary module with simple utility and informative functions.

Link to project repository
--------------------------
`https://github.com/pklesk/mcts_numba_cuda <https://github.com/pklesk/mcts_numba_cuda>`_ 
"""

import cpuinfo
import platform
import psutil
from numba import cuda
import pickle
import time
import zipfile as zf
import os
import json
import sys
 
__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"

def dict_to_str(d, indent=0):
    """Returns a vertically formatted string representation of a dictionary."""
    indent_str = indent * " "
    dict_str = indent_str + "{"
    for i, key in enumerate(d):
        dict_str += "\n" + indent_str + "  "  + str(key) + ": " + str(d[key]) + ("," if i < len(d) - 1 else "")    
    dict_str += "\n" + indent_str + "}"
    return dict_str

def list_to_str(l, indent=0):
    """Returns a vertically formatted string representation of a list."""
    indent_str = indent * " "
    list_str = ""
    for i, elem in enumerate(l):
        list_str += indent_str
        list_str += "[" if i == 0 else " "  
        list_str += str(elem) + (",\n" if i < len(l) - 1 else "]")
    return list_str 

def pickle_objects(fname, some_list):
    """Pickles a list of objects to a binary file."""
    print(f"PICKLE OBJECTS... [to file: {fname}]")
    t1 = time.time()
    try:
        f = open(fname, "wb+")
        pickle.dump(some_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
    except IOError:
        sys.exit("[error occurred when trying to open or pickle the file]")
    t2 = time.time()
    print(f"PICKLE OBJECTS DONE. [time: {t2 - t1} s]")

def unpickle_objects(fname):
    """Returns an a list of objects from a binary file."""
    print(f"UNPICKLE OBJECTS... [from file: {fname}]")
    t1 = time.time()
    try:    
        f = open(fname, "rb")
        some_list = pickle.load(f)
        f.close()
    except IOError:
        sys.exit("[error occurred when trying to open or read the file]")
    t2 = time.time()
    print(f"UNPICKLE OBJECTS DONE. [time: {t2 - t1} s]")
    return some_list

def cpu_and_system_props():
    """Returns a dictionary with properties of CPU and OS."""
    props = {}    
    info = cpuinfo.get_cpu_info()
    un = platform.uname()
    props["cpu_name"] = info["brand_raw"]
    props["ram_size"] = f"{psutil.virtual_memory().total / 1024**3:.1f} GB"
    props["os_name"] = f"{un.system} {un.release}"
    props["os_version"] = f"{un.version}"
    props["os_machine"] = f"{un.machine}"    
    return props    

def gpu_props():
    """Returns a dictionary with properties of GPU device."""
    gpu = cuda.get_current_device()
    props = {}
    props["name"] = gpu.name.decode("ASCII")
    props["max_threads_per_block"] = gpu.MAX_THREADS_PER_BLOCK
    props["max_block_dim_x"] = gpu.MAX_BLOCK_DIM_X
    props["max_block_dim_y"] = gpu.MAX_BLOCK_DIM_Y
    props["max_block_dim_z"] = gpu.MAX_BLOCK_DIM_Z
    props["max_grid_dim_x"] = gpu.MAX_GRID_DIM_X
    props["max_grid_dim_y"] = gpu.MAX_GRID_DIM_Y
    props["max_grid_dim_z"] = gpu.MAX_GRID_DIM_Z    
    props["max_shared_memory_per_block"] = gpu.MAX_SHARED_MEMORY_PER_BLOCK
    props["async_engine_count"] = gpu.ASYNC_ENGINE_COUNT
    props["can_map_host_memory"] = gpu.CAN_MAP_HOST_MEMORY
    props["multiprocessor_count"] = gpu.MULTIPROCESSOR_COUNT
    props["warp_size"] = gpu.WARP_SIZE
    props["unified_addressing"] = gpu.UNIFIED_ADDRESSING
    props["pci_bus_id"] = gpu.PCI_BUS_ID
    props["pci_device_id"] = gpu.PCI_DEVICE_ID
    props["compute_capability"] = gpu.compute_capability            
    CC_CORES_PER_SM_DICT = {
        (2,0) : 32,
        (2,1) : 48,
        (3,0) : 256,
        (3,5) : 256,
        (3,7) : 256,
        (5,0) : 128,
        (5,2) : 128,
        (6,0) : 64,
        (6,1) : 128,
        (7,0) : 64,
        (7,5) : 64,
        (8,0) : 64,
        (8,6) : 128
        }
    props["cores_per_SM"] = CC_CORES_PER_SM_DICT.get(gpu.compute_capability)
    props["cores_total"] = props["cores_per_SM"] * gpu.MULTIPROCESSOR_COUNT
    return props

def hash_function(s):
    """Returns a hash code (integer) for given string as a base 31 expansion."""
    h = 0
    for c in s:
        h *= 31 
        h += ord(c)
    return h

def hash_str(params, digits):
    return str((hash_function(str(params)) & ((1 << 32) - 1)) % 10**digits).rjust(digits, "0") 

class Logger:
    """Class for simultaneous logging to console and a log file (for purposes of experiments)."""
    def __init__(self, fname):
        """Constructor of ``MCTSNC`` instances."""
        self.logfile = open(fname, "a", encoding="utf-8")  
        
    def write(self, message):
        """Writes a message to console and a log file.""" 
        self.logfile.write(message)
        self.logfile.flush() 
        sys.__stdout__.write(message)

    def flush(self):
        """Empty function required for buffering."""
        pass  
    
def experiment_hash_str(matchup_info, c_props, g_props, main_hs_digits=10, matchup_hs_digits=5, env_hs_digits=3):
    """Returns a hash string for an experiment, based on its settings and properties."""
    matchup_hs = hash_str(matchup_info, digits=matchup_hs_digits)
    env_props = {**c_props, **g_props}    
    env_hs =  hash_str(env_props, digits=env_hs_digits)
    all_info = {**matchup_info, **env_props}
    all_hs = hash_str(all_info, digits=main_hs_digits)
    hs = f"{all_hs}_{matchup_hs}_{env_hs}_[{matchup_info['ai_a_shortname']};{matchup_info['ai_b_shortname']};{matchup_info['game_name']};{matchup_info['n_games']}]"
    return hs

def save_and_zip_experiment(experiment_hs, experiment_info, folder):
    """Saves and zips .json and .log files for an experiment given its hash string and information stored in a dictionary."""
    print(f"SAVE AND ZIP EXPERIMENT... [hash string: {experiment_hs}]")
    t1 = time.time()
    fpath = folder + experiment_hs    
    try:        
        f = open(fpath + ".json", "w+")
        json.dump(experiment_info, f, indent=2)
        f.close()
        with zf.ZipFile(fpath + ".zip", mode="w", compression=zf.ZIP_DEFLATED) as archive:
                archive.write(fpath + ".json", arcname=experiment_hs + ".json")
                archive.write(fpath + ".log", arcname=experiment_hs + ".log")
        os.remove(fpath + ".json")
        os.remove(fpath + ".log") 
    except IOError:
        sys.exit(f"[error occurred when trying to save and zip experiment info: {fname}]")            
    t2 = time.time()
    print(f"SAVE AND ZIP EXPERIMENT DONE. [time: {t2 - t1} s]")

def unzip_and_load_experiment(experiment_hs, folder):
    """Unzips, loads an experiment given its hash string, and returns a dictionary with experiments' information."""
    print(f"UNZIP AND LOAD EXPERIMENT... [hash string: {experiment_hs}]")
    t1 = time.time()
    fpath = folder + experiment_hs    
    try:        
        with zf.ZipFile(fpath + ".zip", "r") as zip_ref:
            zip_ref.extract(experiment_hs + ".json", path=os.path.dirname(fpath + ".json"))            
        with open(fpath + ".json", 'r', encoding="utf-8") as json_file:
            experiment_info = json.load(json_file) 
        os.remove(fpath + ".json") # TODO uncomment this back, to have extracted file removed once used
    except IOError:
        sys.exit(f"[error occurred when trying to unzip and load experiment info: {experiment_hs}]")            
    t2 = time.time()
    print(f"UNZIP AND LOAD EXPERIMENT DONE. [time: {t2 - t1} s]")
    return experiment_info