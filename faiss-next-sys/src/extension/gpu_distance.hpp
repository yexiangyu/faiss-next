#pragma once
#include <cstdint>

typedef void GpuDistanceParams;
typedef void GpuResources;
// typedef void GpuVectors;

int gpu_distance_params_new(GpuDistanceParams **params, int metric, int dims, int device);
void gpu_distance_params_free(GpuDistanceParams *params);
void gpu_distance_params_get_dims(GpuDistanceParams *params, int *dims);
void gpu_distance_params_get_k(GpuDistanceParams *params, int *k);
void gpu_distance_params_set_k(GpuDistanceParams *params, int k);
void gpu_distance_params_set_vectors(GpuDistanceParams *params, const float *vectors);
void gpu_distance_params_set_num_vectors(GpuDistanceParams *params, int num_vectors);
void gpu_distance_params_get_num_vectors(GpuDistanceParams *params, int *num_vectors);
void gpu_distance_params_set_queries(GpuDistanceParams *params, const float *queries);
void gpu_distance_params_set_num_queries(GpuDistanceParams *params, int num_queries);
void gpu_distance_params_get_num_queries(GpuDistanceParams *params, int *num_queries);
void gpu_distance_params_get_results(GpuDistanceParams *params, float **results);
void gpu_distance_params_set_results(GpuDistanceParams *params, float *results);
void gpu_distance_params_get_indices(GpuDistanceParams *params, int64_t **indices);
void gpu_distance_params_set_indices(GpuDistanceParams *params, int64_t *indices);
int gpu_distance_params_compute(GpuDistanceParams *params, GpuResources *resources);
