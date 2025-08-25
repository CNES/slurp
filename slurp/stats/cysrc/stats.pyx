# distutils: language = c++
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of SLURP
# (see https://github.com/CNES/slurp).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    
    void finalize_seg(unsigned int * , unsigned int * , unsigned int * , unsigned int , unsigned int)


# End PXD

# Create a Cython extension 
# and create a bunch of forwarding methods
# Python extension type.
cdef class PyStats:

    def __cinit__(self):
        pass

    
    def run_stats(self, primitives: np.ndarray, labelImage: np.ndarray, nbLabels):
        # Compute sums of different primitives, for each labeled segments of an image
        # returns an accumulator and a counter 
        # accumulator : sum of each primitive for each segment (dimension : nbBands * nbLabels)
        # counter : nb of pixels by segment (dimension : nbLabels)
        
        nbBands = primitives.shape[0]
        nbRows = primitives.shape[1]
        nbCols = primitives.shape[2]

        cdef float[::1] primitives_memview = primitives.flatten().astype(np.float32)
        cdef unsigned int[::1] label_img_memview = labelImage.flatten().astype(np.uint32)
	
        cdef float[::1] accumulator_mem_view = np.zeros(nbLabels*nbBands).astype(np.float32)
        cdef unsigned int[::1] counter_mem_view = np.zeros(nbLabels).astype(np.uint32)
        
        compute_stats(&primitives_memview[0],
                      &label_img_memview[0], 
                      &accumulator_mem_view[0], 
                      &counter_mem_view[0],
                      nbLabels, 
                      nbBands, 
                      nbRows, 
                      nbCols)

        return np.asarray(accumulator_mem_view), np.asarray(counter_mem_view)

    def finalize(self, segmentation: np.ndarray, clustering: np.ndarray):
        # Takes a segmented image as input, a clustering (labels -> class)
        # and returns a classification map (each segment of the image defined by a class)
                
        nbRows = segmentation.shape[1]
        nbCols = segmentation.shape[2]
        
        cdef unsigned int[::1] seg_memview = segmentation.flatten().astype(np.uint32) 
        cdef unsigned int[::1] cluster_memview = clustering.flatten().astype(np.uint32)
        
        cdef unsigned int[::1] final_image_memview = np.zeros(nbRows*nbCols).astype(np.uint32)
        
        finalize_seg(&seg_memview[0], &cluster_memview[0], &final_image_memview[0], nbRows, nbCols)
        
        return np.asarray(final_image_memview).reshape((nbRows, nbCols))
        
    
