#ifndef _SLLN_HPP_
#define _SLLN_HPP_
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// CUDA utilities and system includes
#include <cuda_runtime.h>

#include <iostream>
#include <string>

extern "C" void applySLLN(const float3 &input, float3 &output, int block_size,
                              int width, int height, float ill, float noise);
extern "C" void initSLLN(int width, int height);
extern "C" void endSLLN();

class SLLN {
 public:
  SLLN();
  ~SLLN();
  bool apply(const cv::Mat& in_image, cv::Mat& out_image,
                                  float illum, float noise);
  bool init(int width, int height);

 private:
  int block_size;
};

#endif  // #ifndef _SLLN_HPP_
