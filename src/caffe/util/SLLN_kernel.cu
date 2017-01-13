#ifndef _SLLN_KERNEL_CH_
#define _SLLN_KERNEL_CH_

#include <helper_functions.h>
#include <helper_math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <curand_kernel.h>
#include <curand_normal.h>
#include <cuda_runtime.h>

#include <math.h>
#include <string>
#include <typeinfo>
#include <vector>


////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host
// call returns an error
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line,
                (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void remosaic(float3 *id, float *od, int width, int height) {
    // assume rggb bayer pattern

    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if ((xIndex >= width) || (yIndex >= height)) return;

    const int tid = yIndex + xIndex * height;

    int i = (2-((xIndex % 2) + (yIndex % 2)));

    switch (i) {
      case 0:
        od[yIndex + xIndex * height] = id[tid].x;
        break;
      case 1:
        od[yIndex + xIndex * height] = id[tid].y;
        break;
      case 2:
        od[yIndex + xIndex * height] = id[tid].z;
        break;
    }
}

__global__ void apply_slln(float *id, float *od, float *numbers, int width, int height,
                                    float ill_mult, float noise) {
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if ((xIndex >= width) || (yIndex >= height)) return;

    int ind = yIndex + xIndex * height;

    curandState state;

    curand_init(clock64(), ind, 0, &state);

    numbers[ind] = curand_normal(&state);

    od[ind] =
            min(1.0f,max(0.0f,id[ind] * ill_mult +
                        numbers[ind] * noise));
}

__global__ void demosaic(float *id, float3 *od, int width, int height) {
  // id is single channel image
  // od is 3-channel RGB image

  const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

  if ((xIndex >= width) || (yIndex >= height)) return;

  int o_i_j = yIndex + height*xIndex;
  int o_1i_j = (yIndex-1) + height*xIndex;
  int o_i1_j = (yIndex+1) + height*xIndex;
  int o_2i_j = (yIndex-2) + height*xIndex;
  int o_i2_j = (yIndex+2) + height*xIndex;
  int o_i_1j = yIndex + height*(xIndex-1);
  int o_i_j1 = yIndex + height*(xIndex+1);
  int o_i_2j = yIndex + height*(xIndex-2);
  int o_i_j2 = yIndex + height*(xIndex+2);
  int o_1i_1j = (yIndex-1) + height*(xIndex-1);
  int o_i1_1j = (yIndex+1) + height*(xIndex-1);
  int o_1i_j1 = (yIndex-1) + height*(xIndex+1);
  int o_i1_j1 = (yIndex+1) + height*(xIndex+1);

  int _1y = (yIndex-1) >= 0;
  int y1 = (yIndex+1) < height;
  int _2y = (yIndex-2) >= 0;
  int y2 = (yIndex+2) < height;
  int _1x = (xIndex-1) >= 0;
  int x1 = (xIndex+1) < width;
  int _2x = (xIndex-2) >= 0;
  int x2 = (xIndex+2) < width;

  float id_i_j = id[o_i_j];
  float id_1i_j = id[o_1i_j] * _1y;
  float id_i_1j = id[o_i_1j] * _1x;
  float id_i1_j = id[o_i1_j] * y1;
  float id_i_j1 = id[o_i_j1] * x1;
  float id_2i_j = id[o_2i_j] * _2y;
  float id_i_2j = id[o_i_2j] * _2x;
  float id_i2_j = id[o_i2_j] * y2;
  float id_i_j2 = id[o_i_j2] * x2;
  float id_1i_1j = id[o_1i_1j] * (_1y & _1x);
  float id_i1_1j = id[o_i1_1j] * (y1 & _1x);
  float id_1i_j1 = id[o_1i_j1] * (_1y & x1);
  float id_i1_j1 = id[o_i1_j1] * (y1 & x1);

  const float Fij = id_i_j;

  //symmetric 4,2,-1 response - cross
  const float R1 = (4*id_i_j + 2*(id_1i_j + id_i_1j + id_i1_j + id_i_j1) - id_2i_j - id_i2_j - id_i_2j - id_i_j2) / (4 + 2*(_1y + _1x + y1 + x1) - _2y - y2 - _2x - x2);

  //left-right symmetric response - with .5 + height*1 + height*4 + height*5 - theta
  const float R2 = (
     8*(id_1i_j + id_i1_j)
    +10*id_i_j
    + id_i_2j + id_i_j2
    - 2*((id_1i_1j + id_i1_1j + id_1i_j1 + id_i1_j1) + id_2i_j + id_i2_j)
  ) / (8*(_1y + y1)
 +10
 + _2x + x2
 - 2*(((_1y & _1x) + (y1 & _1x) + (_1y & x1) + (y1 & x1)) + _2y + y2));

  //top-bottom symmetric response - with .5 + height*1 + height*4 + height*5 - phi
  const float R3 = (
      8*(id_i_1j + id_i_j1)
     +10*id_i_j
     + id_2i_j + id_i2_j
     - 2*((id_1i_1j + id_i1_1j + id_1i_j1 + id_i1_j1) + id_i_2j + id_i_j2)
  ) / (8*(_1x + x1)
  +10
  + _2y + y2
  - 2*(((_1y & _1x) + (y1 & _1x) + (_1y & x1) + (y1 & x1)) + _2x + x2));



  //symmetric 3/2s response - checker
  const float R4 = (
       12*id_i_j
      - 3*(id_2i_j + id_i2_j + id_i_2j + id_i_j2)
      + 4*(id_1i_1j + id_i1_1j + id_1i_j1 + id_i1_j1)
  ) / (12
 - 3*(_2y + y2 + _2x + x2)
 + 4*((_1y & _1x) + (y1 & _1x) + (_1y & x1) + (y1 & x1)));


  const float G_at_red_or_blue = R1;
  const float R_at_G_in_red = R2;
  const float B_at_G_in_blue = R2;
  const float R_at_G_in_blue = R3;
  const float B_at_G_in_red = R3;
  const float R_at_B = R4;
  const float B_at_R = R4;

  //RGGB -> RedXY = (0, 0), GreenXY1 = (1, 0), GreenXY2 = (0, 1), BlueXY = (1, 1)
  //GRBG -> RedXY = (1, 0), GreenXY1 = (0, 0), GreenXY2 = (1, 1), BlueXY = (0, 1)
  //GBRG -> RedXY = (0, 1), GreenXY1 = (0, 0), GreenXY2 = (1, 1), BlueXY = (1, 0)
  //BGGR -> RedXY = (1, 1), GreenXY1 = (1, 0), GreenXY2 = (0, 1), BlueXY = (0, 0)
  const int r_mod_2 = xIndex & 1;
  const int c_mod_2 = yIndex & 1;
  #define is_rggb (true)
  #define is_grbg (false)
  #define is_gbrg (false)
  #define is_bggr (false)

  const int red_col = is_grbg | is_bggr;
  const int red_row = is_gbrg | is_bggr;
  const int blue_col = 1 - red_col;
  const int blue_row = 1 - red_row;

  const int in_red_row = r_mod_2 == red_row;
  const int in_blue_row = r_mod_2 == blue_row;
  const int is_red_pixel = (r_mod_2 == red_row) & (c_mod_2 == red_col);
  const int is_blue_pixel = (r_mod_2 == blue_row) & (c_mod_2 == blue_col);
  const int is_green_pixel = !(is_red_pixel | is_blue_pixel);

  assert(is_green_pixel + is_blue_pixel + is_red_pixel == 1);
  assert(in_red_row + in_blue_row == 1);

  //at R locations: R is original
  //at B locations it is the 3/2s symmetric response
  //at G in red rows it is the left-right symmmetric with 4s
  //at G in blue rows it is the top-bottom symmetric with 4s
  float red =
      Fij * is_red_pixel +
      R_at_B * is_blue_pixel +
      R_at_G_in_red * (is_green_pixel & in_red_row) +
      R_at_G_in_blue * (is_green_pixel & in_blue_row);

  //at B locations: B is original
  //at R locations it is the 3/2s symmetric response
  //at G in red rows it is the top-bottom symmmetric with 4s
  //at G in blue rows it is the left-right symmetric with 4s
  float blue =
      Fij * is_blue_pixel +
      B_at_R * is_red_pixel +
      B_at_G_in_red * (is_green_pixel & in_red_row) +
      B_at_G_in_blue * (is_green_pixel & in_blue_row);

  //at G locations: G is original
  //at R locations: symmetric 4,2,-1
  //at B locations: symmetric 4,2,-1
  float green = Fij * is_green_pixel + G_at_red_or_blue * (!is_green_pixel);

  od[o_i_j].x = blue;
  od[o_i_j].y = green;
  od[o_i_j].z = red;
}

extern "C" void applySLLN(float3 &input, float3 &output, int block_size, int width, int height, float ill, float noise) {

    float *d_gray_remosaic, *d_gray_slln, *d_gray_noise;

    float3 *d_input, *d_output;

    const int colorBytes = width * sizeof(float3) * height;

    const int grayBytes = width * sizeof(float) * height;

    checkCudaErrors(cudaMalloc((void **)&d_input, colorBytes));
    checkCudaErrors(cudaMalloc((void **)&d_output, colorBytes));
    checkCudaErrors(cudaMalloc((void **)&d_gray_remosaic, grayBytes));
    checkCudaErrors(cudaMalloc((void **)&d_gray_slln, grayBytes));
    checkCudaErrors(cudaMalloc((void **)&d_gray_noise, grayBytes));

    // Copy data from OpenCV input image to device memory
    checkCudaErrors(
        cudaMemcpy(d_input, &input, colorBytes, cudaMemcpyHostToDevice));

    // Specify a reasonable block size
    const dim3 block(block_size, block_size);


    const dim3 grid((width + block.x - 1) / block.x,
                    (height + block.y - 1) / block.y);

    remosaic<<<grid, block>>>(d_input, d_gray_remosaic, width,
                                        height);

    // Synchronize to check for any kernel launch errors
    checkCudaErrors(cudaDeviceSynchronize());

    apply_slln<<<grid, block>>>(d_gray_remosaic, d_gray_slln, d_gray_noise,
                                width, height, ill, noise);

    checkCudaErrors(cudaDeviceSynchronize());

    demosaic<<<grid, block>>>(d_gray_slln, d_output, width,
                                height);

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(&output, d_output, colorBytes, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
    checkCudaErrors(cudaFree(d_gray_remosaic));
    checkCudaErrors(cudaFree(d_gray_slln));
    checkCudaErrors(cudaFree(d_gray_noise));

    return;
}

#endif  // #ifndef _SEQSLAM_KERNEL_H_
