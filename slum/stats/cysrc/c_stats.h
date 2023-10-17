#ifndef C_STATS_H
#define C_STATS_H
#include <iostream>

namespace stats {

  void compute_stats(float * color_img, unsigned int * label_img, 
		     float * accumulator, unsigned int * counter, 
		     unsigned int num_labels, unsigned int nb_bands,
		     unsigned int nb_rows, unsigned int nb_cols);

  void compute_stats_mb(float * color_img, unsigned int * label_img, 
		     float * accumulator, unsigned int * counter, 
		     unsigned int num_labels, unsigned int nb_bands,
		     unsigned int nb_rows, unsigned int nb_cols);

} // end of namespace turbostats

#endif
