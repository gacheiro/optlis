from ctypes import *


class c_task(Structure):
    _fields_ = [
        ("type", c_int32),
        ("site", c_int32),
        ("target", c_int32),
    ]


class c_instance(Structure):
    _fields_ = [
        ("nnodes", c_size_t),
        ("ntasks", c_size_t),
        ("nproducts", c_size_t),
        ("ntime_units", c_size_t),
        ("nresources", POINTER(c_int32)),
        ("nodes", POINTER(c_int32)),
        ("cleaning_start_times", POINTER(c_int32)),
        ("neutralizing_start_times", POINTER(c_int32)),
        ("products_risk", POINTER(c_double)),
        ("degradation_rates", POINTER(c_double)),
        ("metabolizing_rates", POINTER(c_double)),
    ]


class c_solution(Structure):
    _fields_ = [
        ("ntasks", c_size_t),
        ("task_list", POINTER(c_task)),
        ("nodes_concentration", POINTER(c_double)),
        ("objective", c_double),
        ("found_at", c_int32),
    ]


class c_budget(Structure):
    _fields_ = [("max", c_int32), ("consumed", c_int32)]
