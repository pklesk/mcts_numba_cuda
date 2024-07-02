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
import sys
import time
 
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