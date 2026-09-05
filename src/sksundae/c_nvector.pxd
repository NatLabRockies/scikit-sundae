from .c_sundials cimport *  # Access to types

# nvector_serial.h
cdef extern from "nvector/nvector_serial.h":    
    N_Vector N_VNew_Serial(sunindextype vec_length, SUNContext ctx)