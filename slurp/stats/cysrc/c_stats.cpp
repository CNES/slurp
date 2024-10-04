/**
* Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
*
* This file is part of SLURP
* (see https://github.com/CNES/slurp).
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include "c_stats.h"
#include <iostream>
using namespace std;
 
namespace stats {

  void compute_stats(float * color_img, unsigned int * label_img, 
		     float * accumulator, unsigned int * counter, 
		     unsigned int num_labels, unsigned int nb_bands,
		     unsigned int nb_rows, unsigned int nb_cols) 
  {
        
    /*
      color_img and label_img are flattened chunks (1D)
      accumulator and counter are already initialized in stats.pyx : their size is num_labels
      num_labels : nb of labels on the whole image
      nb_bands, nb_rows and nb_cols : shape of the chunk
    */
    
    unsigned int label_coords, seg, index_seg;
    unsigned int num_pixels = nb_rows * nb_cols;

    for(unsigned int r = 0; r < nb_rows; r++){
      for(unsigned int c = 0; c < nb_cols; c++){
	// label_coors : rank in the flattened array
        label_coords = r * nb_cols + c;
	seg = label_img[label_coords];
	// first segment is 1 ; index_seg is 0
	index_seg = seg - 1;
        if (index_seg != -1){
	  counter[index_seg]++;
	  for(unsigned int b = 0; b < nb_bands; b++){
	    accumulator[ b * num_labels + index_seg ] += color_img[ num_pixels * b + label_coords ];
	  }
        }
      }
    }
    // accumulator contains the sum of each band for each label
    // counter contains the nb of occurrence (pixel) for each label
  }


  void finalize_seg(unsigned int * segmentation, unsigned int * clustering, 
                    unsigned int * final_image, unsigned int nb_rows, unsigned int nb_cols) 
  {
    /*
      segmentation is the flattened segmented image containing nbLabels (dimension nb_rows * nb_cols)
      clustering contains a class for each segment (dimension : nbLabels)
      this method computes final_image and associates for each segment the corresponding class
    */
    unsigned int label_coords, seg, seg_class;
      
    for (unsigned r = 0; r < nb_rows; r++) {
         for(unsigned int c = 0; c < nb_cols; c++){
             label_coords = r * nb_cols + c;
             seg = segmentation[label_coords];
             if (seg != 0) {
	       seg_class = clustering[seg - 1];
	       final_image[label_coords] = seg_class;
             }
        }
    }
    
  }
}
