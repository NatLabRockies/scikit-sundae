from .c_sundials cimport *  # Access to types

# sunmatrix_dense.h
cdef extern from "sunmatrix/sunmatrix_dense.h":
    SUNMatrix SUNDenseMatrix(sunindextype M, sunindextype N, SUNContext ctx)

    sunrealtype** SUNDenseMatrix_Cols(SUNMatrix A)

# sunmatrix_band.h
cdef extern from "sunmatrix/sunmatrix_band.h":
    SUNMatrix SUNBandMatrix(sunindextype M, sunindextype mu, sunindextype ml,
                            SUNContext ctx)

    sunindextype SUNBandMatrix_StoredUpperBandwidth(SUNMatrix A)
    sunindextype SUNBandMatrix_LowerBandwidth(SUNMatrix A)
    sunindextype SUNBandMatrix_UpperBandwidth(SUNMatrix A)
    sunrealtype** SUNBandMatrix_Cols(SUNMatrix A)

# sunmatrix_sparse.h
cdef extern from "sunmatrix/sunmatrix_sparse.h":
    int CSC_MAT
    int CSR_MAT

    SUNMatrix SUNSparseMatrix(sunindextype M, sunindextype N, sunindextype NNZ,
                              int sparsetype, SUNContext ctx)

    sunrealtype* SUNSparseMatrix_Data(SUNMatrix A)
    sunindextype* SUNSparseMatrix_IndexValues(SUNMatrix A)
    sunindextype* SUNSparseMatrix_IndexPointers(SUNMatrix A)