#ifdef USE_CUDA

#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/StandardGpuResources.h>
#include "gpu_distance.hpp"

struct GpuDistanceParamsInternal {
    faiss::gpu::GpuDistanceParams params;
    int device;
};

int gpu_distance_params_new(GpuDistanceParams **params, int metric, int dims, int device)
{
    try
    {
        auto *internal = new GpuDistanceParamsInternal();
        internal->device = device;
        internal->params.metric = static_cast<faiss::MetricType>(metric);
        internal->params.dims = dims;
        *params = (GpuDistanceParams *)internal;
        return 0;
    }
    catch (std::exception &e)
    {
        return -1;
    }
}

void gpu_distance_params_free(GpuDistanceParams *params)
{
    delete (GpuDistanceParamsInternal *)params;
}

void gpu_distance_params_get_dims(GpuDistanceParams *params, int *dims)
{
    auto *internal = (GpuDistanceParamsInternal *)params;
    *dims = internal->params.dims;
}

void gpu_distance_params_set_vectors(GpuDistanceParams *params, const float *vectors)
{
    auto *internal = (GpuDistanceParamsInternal *)params;
    internal->params.vectors = vectors;
}

void gpu_distance_params_get_k(GpuDistanceParams *params, int *k)
{
    auto *internal = (GpuDistanceParamsInternal *)params;
    *k = internal->params.k;
}

void gpu_distance_params_set_k(GpuDistanceParams *params, int k)
{
    auto *internal = (GpuDistanceParamsInternal *)params;
    internal->params.k = k;
}

void gpu_distance_params_get_num_vectors(GpuDistanceParams *params, int *num_vectors)
{
    auto *internal = (GpuDistanceParamsInternal *)params;
    *num_vectors = internal->params.numVectors;
}

void gpu_distance_params_get_num_queries(GpuDistanceParams *params, int *num_queries)
{
    auto *internal = (GpuDistanceParamsInternal *)params;
    *num_queries = internal->params.numQueries;
}

void gpu_distance_params_set_num_vectors(GpuDistanceParams *params, int num_vectors)
{
    auto *internal = (GpuDistanceParamsInternal *)params;
    internal->params.numVectors = num_vectors;
}

void gpu_distance_params_set_queries(GpuDistanceParams *params, const float *queries)
{
    auto *internal = (GpuDistanceParamsInternal *)params;
    internal->params.queries = queries;
}

void gpu_distance_params_set_num_queries(GpuDistanceParams *params, int num_queries)
{
    auto *internal = (GpuDistanceParamsInternal *)params;
    internal->params.numQueries = num_queries;
}

void gpu_distance_params_get_results(GpuDistanceParams *params, float **results)
{
    auto *internal = (GpuDistanceParamsInternal *)params;
    *results = internal->params.outDistances;
}

void gpu_distance_params_set_results(GpuDistanceParams *params, float *results)
{
    auto *internal = (GpuDistanceParamsInternal *)params;
    internal->params.outDistances = results;
}

void gpu_distance_params_get_indices(GpuDistanceParams *params, int64_t **indices)
{
    auto *internal = (GpuDistanceParamsInternal *)params;
    *indices = internal->params.outIndices;
}

void gpu_distance_params_set_indices(GpuDistanceParams *params, int64_t *indices)
{
    auto *internal = (GpuDistanceParamsInternal *)params;
    internal->params.outIndices = indices;
}

int gpu_distance_params_compute(GpuDistanceParams *params, GpuResources *resources)
{
    try
    {
        auto *internal = (GpuDistanceParamsInternal *)params;
        auto *res = (faiss::gpu::GpuResources *)resources;
        faiss::gpu::bfKnn(*res, internal->device, internal->params);
        return 0;
    }
    catch (std::exception &e)
    {
        return -1;
    }
}

#endif