#pragma once
#include <cstddef>

extern "C" {
const char* hpc_error();
int hpc_has_openmp();
int hpc_search(const float* queries, const float* database, int nq, int n, int dim,
               int k, int chunk, int threads, int parallel, int* indices, float* scores);
int hpc_descriptors(const unsigned char* images, int n, int height, int width,
                    int bins, int threads, int parallel, float* output);
int hpc_inverted_index(const int* labels, int n, int clusters, int* offsets, int* permutation);
const char* hpc_cuda_error();
int hpc_cuda_devices();
int hpc_cuda_set_device(int device);
int hpc_cuda_search(const float*, const float*, int, int, int, int, int, int, int*, float*);
int hpc_cuda_descriptors(const unsigned char*, int, int, int, int, int, float*);
int hpc_cuda_inverted_index(const int*, int, int, int*, int*);
}
