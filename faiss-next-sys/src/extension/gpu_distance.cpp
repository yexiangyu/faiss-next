#include "gpu_distance.hpp"
#include "faiss/gpu/GpuDistance.h"
#include <cstring>

int gpu_distance_params_new(GpuDistanceParams **params, int metric, int dims, int device)
{
    try
    {
        auto p = new faiss::gpu::GpuDistanceParams();
        p->metric = static_cast<faiss::MetricType>(metric);
        p->dims = dims;
        p->device = device;
        p->k = -1;
        *params = p;
        return 0;
    }
    catch (std::exception &e)
    {
        return 999999999;
    }
}

void gpu_distance_params_free(GpuDistanceParams *params)
{
    auto p = (faiss::gpu::GpuDistanceParams *)params;

    if (p->outDistances != nullptr)
    {
        free(p->outDistances);
    }

    if (p->outIndices != nullptr)
    {
        free(p->outIndices);
    }
    delete p;
}

void gpu_distance_params_get_dims(GpuDistanceParams *params, int *dims)
{
    *dims = ((faiss::gpu::GpuDistanceParams *)params)->dims;
}

void gpu_distance_params_get_k(GpuDistanceParams *params, int *k)
{
    *k = ((faiss::gpu::GpuDistanceParams *)params)->k;
}

void gpu_distance_params_set_k(GpuDistanceParams *params, int k)
{
    ((faiss::gpu::GpuDistanceParams *)params)->k = k;
}

void gpu_distance_params_get_num_vectors(GpuDistanceParams *params, int *num_vectors)
{
    *num_vectors = ((faiss::gpu::GpuDistanceParams *)params)->numVectors;
}

void gpu_distance_params_get_metric(GpuDistanceParams *params, int *metric)
{
    *metric = ((faiss::gpu::GpuDistanceParams *)params)->metric;
}

void gpu_distance_params_set_vectors(GpuDistanceParams *params, const float *vectors)
{
    ((faiss::gpu::GpuDistanceParams *)params)->vectors = vectors;
}

void gpu_distance_params_set_num_vectors(GpuDistanceParams *params, int num_vectors)
{
    ((faiss::gpu::GpuDistanceParams *)params)->numVectors = num_vectors;
}

void gpu_distance_params_set_queries(GpuDistanceParams *params, const float *queries)
{
    ((faiss::gpu::GpuDistanceParams *)params)->queries = queries;
}

void gpu_distance_params_set_num_queries(GpuDistanceParams *params, int num_queries)
{
    ((faiss::gpu::GpuDistanceParams *)params)->numQueries = num_queries;
}

void gpu_distance_params_get_num_queries(GpuDistanceParams *params, int *num_queries)
{
    *num_queries = ((faiss::gpu::GpuDistanceParams *)params)->numQueries;
}

void gpu_distance_params_get_results(GpuDistanceParams *params, float **results)
{
    *results = ((faiss::gpu::GpuDistanceParams *)params)->outDistances;
}

void gpu_distance_params_get_indices(GpuDistanceParams *params, int64_t **indices)
{
    *indices = (int64_t *)((faiss::gpu::GpuDistanceParams *)params)->outIndices;
}

void gpu_distance_params_set_results(GpuDistanceParams *params, float *results)
{
    ((faiss::gpu::GpuDistanceParams *)params)->outDistances = results;
}

void gpu_distance_params_set_indices(GpuDistanceParams *params, int64_t *indices)
{
    ((faiss::gpu::GpuDistanceParams *)params)->outIndices = indices;
}

int gpu_distance_params_compute(GpuDistanceParams *params, GpuResources *resources)
{
    auto p = (faiss::gpu::GpuDistanceParams *)params;
    if (p->numVectors <= 0 || p->numQueries <= 0 || p->k == 0)
        return 999999999;

    auto num_results = p->numVectors * p->numQueries;

    if (p->k > 0)
    {
        num_results = p->k * p->numQueries;
    }

    if (p->outDistances == nullptr)
    {
        p->outDistances = (float *)malloc(sizeof(float) * num_results);
    }
    else
    {
        p->outDistances = (float *)realloc(p->outDistances, sizeof(float) * num_results);
    }

    if (p->outIndices == nullptr)
    {
        p->outIndices = (int64_t *)malloc(sizeof(int64_t) * num_results);
    }
    else
    {
        p->outIndices = (int64_t *)realloc(p->outIndices, sizeof(int64_t) * num_results);
    }

    std::memset(p->outDistances, 0, sizeof(float) * num_results);
    std::memset(p->outIndices, 0, sizeof(int64_t) * num_results);

    faiss::gpu::bfKnn((faiss::gpu::GpuResourcesProvider *)resources, *(faiss::gpu::GpuDistanceParams *)params);

    return 0;
}