# distutils: language = c++
# coding: utf8

import numpy as np
#cimport numpy as np

def npAsContiguousArray(arr : np.array) -> np.array:
    """
    This method checks that the input array is contiguous. 
    If not, returns the contiguous version of the input numpy array.

    Args:
        arr: input array.

    Returns:
        contiguous array usable in C++.
    """
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    return arr

# Begin PXD

# Necessary to include the C++ code
cdef extern from "c_stats.cpp":
    pass

# Declare the class with cdef
cdef extern from "c_stats.h" namespace "stats":

    void compute_stats(float * , unsigned int * , 
		     float * , unsigned int * , 
		     unsigned int , unsigned int ,
		     unsigned int , unsigned int )

    void compute_stats_mb(float * , unsigned int * , 
		     float * , unsigned int * , 
		     unsigned int , unsigned int ,
		     unsigned int , unsigned int )


# End PXD

# Create a Cython extension 
# and create a bunch of forwarding methods
# Python extension type.
cdef class PyStats:

    def __cinit__(self):
        pass

    def run_stats(self, colorImage: np.ndarray, labelImage: np.ndarray, nbLabels):
        
        cdef float[::1] color_img_memview = colorImage.flatten().astype(np.float32)
        cdef unsigned int[::1] label_img_memview = labelImage.flatten().astype(np.uint32)
	
        cdef float[::1] accumulator_mem_view = np.zeros(nbLabels).astype(np.float32)
        cdef unsigned int[::1] counter_mem_view = np.zeros(nbLabels).astype(np.uint32)
        
        nbBands = colorImage.shape[0]
        nbRows = colorImage.shape[1]
        nbCols = colorImage.shape[2]

        compute_stats(&color_img_memview[0],
                      &label_img_memview[0], 
                      &accumulator_mem_view[0], 
                      &counter_mem_view[0],
                      nbLabels, 
                      nbBands, 
                      nbRows, 
                      nbCols)

        return np.asarray(accumulator_mem_view), np.asarray(counter_mem_view)
        
    def run_stats_mb(self, primitives: np.ndarray, labelImage: np.ndarray, nbLabels):
        
        nbBands = primitives.shape[0]
        nbRows = primitives.shape[1]
        nbCols = primitives.shape[2]

        cdef float[::1] primitives_memview = primitives.flatten().astype(np.float32)
        cdef unsigned int[::1] label_img_memview = labelImage.flatten().astype(np.uint32)
	
        cdef float[::1] accumulator_mem_view = np.zeros(nbLabels*nbBands).astype(np.float32)
        cdef unsigned int[::1] counter_mem_view = np.zeros(nbLabels).astype(np.uint32)
        
        compute_stats_mb(&primitives_memview[0],
                      &label_img_memview[0], 
                      &accumulator_mem_view[0], 
                      &counter_mem_view[0],
                      nbLabels, 
                      nbBands, 
                      nbRows, 
                      nbCols)

        return np.asarray(accumulator_mem_view), np.asarray(counter_mem_view)
