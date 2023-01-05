from ctypes import *


class c_instance(Structure):
    _fields_ = [
        ("nnodes", c_size_t),
        ("ntasks", c_size_t),
        ("nresources", c_size_t),
        ("tasks", POINTER(c_int32)),
        ("node_durations", POINTER(c_int32)),
        ("node_risks", POINTER(c_double)),
        ("travel_times", POINTER(c_int32)),
    ]


class c_solution(Structure):
    _fields_ = [
        ("ntasks", c_size_t),
        ("task_list", POINTER(c_int32)),
        ("task_risks", POINTER(c_double)),
        ("objective", c_double),
        ("start_times", POINTER(c_int32)),
        ("finish_times", POINTER(c_int32)),
        ("found_at", c_int32),
        ("relaxation_threshold", c_double),
    ]


class c_budget(Structure):
    _fields_ = [("max", c_int32), ("consumed", c_int32)]
