#include "c_stats.h"
#include <iostream>
using namespace std;
 
namespace stats {

  void compute_stats_single_band(float * color_img, unsigned int * label_img, 
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
    
    for(unsigned int r = 0; r < nb_rows; r++){
      for(unsigned int c = 0; c < nb_cols; c++){
        label_coords = r * nb_cols + c;
	seg = label_img[label_coords];
	index_seg = seg - 1;
         /*
         Pixel Image[r][c] -> color_img[label_coords] --> included in segment label_img[label_coords]
         Accumulator[label_img[label_coords]] += value
         Counter[label_img[label_coords]] += 1
         */
        //counter[label_coords]++;
        counter[index_seg]++;
        /*for(unsigned int b = 0; b < nb_bands; b++){
          accumulator[ b * num_labels + label_img[label_coords] ] += color_img[ num_pixels * b + label_coords ];
        }
        */
        // single band version
        accumulator[index_seg] += color_img[label_coords];
      }
    }
    
    for(unsigned int l = 0; l < num_labels; l++){
          /*for(unsigned int b = 0; b < nb_bands; b++){
        accumulator[ b * num_labels + l ] /= counter[l];
          }
          */
        if (counter[l] > 0) {
            accumulator[l] /= counter[l];
        }
    }

    // accumulator should contain the mean value for each band for each label ;)
  }

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
        label_coords = r * nb_cols + c;
            seg = label_img[label_coords];
            index_seg = seg - 1;
         /*
         Pixel Image[r][c] -> color_img[label_coords] --> included in segment label_img[label_coords]
         Accumulator[label_img[label_coords]] += value
         Counter[label_img[label_coords]] += 1
         */
        if (index_seg != -1){
        counter[index_seg]++;
        for(unsigned int b = 0; b < nb_bands; b++){
          accumulator[ b * num_labels + index_seg ] += color_img[ num_pixels * b + label_coords ];
        }
        }
        }
    }
    
    for(unsigned int b = 0; b < nb_bands; b++){
      for(unsigned int l = 0; l < num_labels; l++){
          if (counter[l] > 0) {
              accumulator[ b * num_labels + l ] /= counter[l];
            }
          }  
    }

    // accumulator should contain the mean value for each band for each label ;)
  }


  void finalize_seg(unsigned int * segmentation, unsigned int * clustering, 
                    unsigned int * final_image, unsigned int nb_rows, unsigned int nb_cols) 
  {
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
