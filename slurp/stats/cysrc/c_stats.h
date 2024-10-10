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
#ifndef C_STATS_H
#define C_STATS_H
#include <iostream>

namespace stats {

      void compute_stats(float * color_img, unsigned int * label_img, 
		     float * accumulator, unsigned int * counter, 
		     unsigned int num_labels, unsigned int nb_bands,
		     unsigned int nb_rows, unsigned int nb_cols);
        
      void finalize_seg(unsigned int * segmentation, unsigned int * clustering, 
                        unsigned int * final_image, unsigned int nb_rows, unsigned int nb_cols); 
    
} // end of namespace turbostats

#endif
