// Copyright 2016 James Sergeant
#include "caffe/util/SLLN.hpp"

#include <typeinfo>

SLLN::SLLN() {
};

SLLN::~SLLN() {
  endSLLN();
};

bool SLLN::init(int width, int height) {
  cudaDeviceProp prop;
  int count;
  cudaError err = cudaGetDeviceCount(&count);
  err = cudaGetDeviceProperties(&prop, 0);
  block_size = floor(sqrt(prop.maxThreadsPerBlock));
  initSLLN(width, height);
  return true;
}

bool SLLN::apply(const cv::Mat& in_image, cv::Mat& out_image,
                                float illum, float noise) {
  if ( in_image.empty() ) {return false;}
  in_image.convertTo(out_image, CV_32F, 1/255.0);
  float3 *temp_out;
  temp_out = reinterpret_cast<float3*>(out_image.data);
  applySLLN(*temp_out, *temp_out, block_size, in_image.cols, in_image.rows,
            illum, noise);

  out_image.convertTo(out_image, CV_8UC3, 255.0);

  return true;
}
