// Copyright 2016 James Sergeant
#include "caffe/util/SLLN.hpp"

SLLN::SLLN() {
  cudaDeviceProp prop;
  int count;
  cudaError err = cudaGetDeviceCount(&count);
  err = cudaGetDeviceProperties(&prop, 0);
  block_size = floor(sqrt(prop.maxThreadsPerBlock));
};

SLLN::~SLLN() {
  // freeTextures();
  // cudaDeviceReset();
};

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

// int main() {
//
//   SLLN slln;
  // cv::Mat image = cv::imread("/home/james/datasets/dark_room_dataset/AppleMouseBottle/DSC00002.JPG",CV_LOAD_IMAGE_COLOR);
  // cv::resize(image,image,cv::Size(256,256));
  // image.convertTo(image, CV_32FC3, 1/255.0);
  //
  // cv::namedWindow( "Original image", cv::WINDOW_NORMAL );// Create a window for display.
  // cv::imshow("Original image",image);
  // // cv::waitKey(0);
  //
  // //   std::string str = std::to_string(image.step * image.rows * sizeof(float));
  //   // std::string str = std::to_string(block_size);
  // //      str = std::to_string(image.rows);
  // // std::cout << str << std::endl;
  // // str = std::to_string(sizeof(image));
  // // std::cout << str << std::endl;
  //
  //   cv::Mat out_image(image.size(),image.type());
  // slln.apply(image, out_image, 0.5f, 0.05f);
  // out_image.convertTo(out_image, CV_8UC3, 255.0);
  //
  //   cv::namedWindow( "Original image", cv::WINDOW_NORMAL );// Create a window for display.
  //   cv::imshow("Original image",image);
  //     cv::namedWindow( "SLLN image", cv::WINDOW_NORMAL );// Create a window for display.
  //     cv::imshow("SLLN image",out_image);
  // cv::waitKey(0);
  // cv::destroyAllWindows();
  // out_image.convertTo(out_image, CV_32FC3, 1/255.0);
  // slln.apply(image, out_image, 0.1f, 0.01f);
  // out_image.convertTo(out_image, CV_8UC3, 255.0);
  //
  //   cv::namedWindow( "Original image", cv::WINDOW_NORMAL );// Create a window for display.
  //   cv::imshow("Original image",image);
  //     cv::namedWindow( "SLLN image", cv::WINDOW_NORMAL );// Create a window for display.
  //     cv::imshow("SLLN image",out_image);
  // cv::waitKey(0);
  // cv::destroyAllWindows();
//
//
//     return 0;
// }
