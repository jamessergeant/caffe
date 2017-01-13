#ifndef _SLLN_H_
#define _SLLN_H_
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// CUDA utilities and system includes
#include <cuda_runtime.h>

#include <iostream>
#include <string>

extern "C" void applySLLN(float3 &input, float3 &output, int block_size,
                              int width, int height, float ill, float noise);
extern "C" void initSLLN(int size);
extern "C" void endSLLN();

class SLLN {
 public:
  SLLN();
  ~SLLN();
  bool apply(const cv::Mat& in_image, cv::Mat out_image, float illum,
                        float noise);
  bool init(int size);

 private:
  int block_size;
  // cv::Mat temp_mat;
};

#endif  // #ifndef _SLLN_H_
