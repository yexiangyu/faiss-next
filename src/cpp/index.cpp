#include "faiss/Index.h"
#include "faiss/impl/FaissException.h"
#include "index.hpp"

int32_t version_major()
{
    return FAISS_VERSION_MAJOR;
}

int32_t version_minor()
{
    return FAISS_VERSION_MINOR;
}

int32_t version_patch()
{
    return FAISS_VERSION_PATCH;
}

void index_free(Index *index)
{
	delete (faiss::Index *)index;
}

int index_d(const Index *index)
{
    return ((faiss::Index *)index)->d;
}

int64_t index_ntotal(const Index *index)
{
    return ((faiss::Index *)index)->ntotal;
}

bool index_verbose(const Index *index)
{
    return ((faiss::Index *)index)->verbose;
}

void index_set_verbose(Index *index, bool verbose)
{
    ((faiss::Index *)index)->verbose = verbose;
}

bool index_is_trained(const Index *index)
{
    return ((faiss::Index *)index)->is_trained;
}

int index_metric_type(const Index *index)
{
    return (int)((faiss::Index *)index)->metric_type;
}

float index_metric_arg(const Index *index)
{
    return ((faiss::Index *)index)->metric_arg;
}

void index_train(Index *index, int64_t n, const float *x)
{
    ((faiss::Index *)index)->train(n, x);
}

void index_add(Index *index, int64_t n, const float *x)
{
    ((faiss::Index *)index)->add(n, x);
}

void index_add_with_ids(Index *index, int64_t n, const float *x, const int64_t *xids)
{
    ((faiss::Index *)index)->add_with_ids(n, x, xids);
}

void index_search(const Index *index, int64_t n, const float *x, int64_t k, float *distances, int64_t *labels, const SearchParameters *params)
{
    ((faiss::Index *)index)->search(n, x, k, distances, labels, (faiss::SearchParameters *)params);
}

void index_range_search(const Index *index, int64_t n, const float *x, float radius, int *result, const SearchParameters *params)
{
    ((faiss::Index *)index)->range_search(n, x, radius, (faiss::RangeSearchResult *)result, (faiss::SearchParameters *)params);
}

void index_assign(const Index *index, int64_t n, const float *x, int64_t *labels, int64_t k)
{
    ((faiss::Index *)index)->assign(n, x, labels, k);
}

void index_reset(Index *index)
{
    ((faiss::Index *)index)->reset();
}

size_t index_remove_ids(Index *index, const IDSelector *sel)
{
    return ((faiss::Index *)index)->remove_ids(*(faiss::IDSelector *)sel);
}

void index_reconstruct(Index *index, int64_t key, float *recons)
{
    ((faiss::Index *)index)->reconstruct(key, recons);
}

void index_reconstruct_batch(Index *index, int64_t n, const int64_t *keys, float *recons)
{
    ((faiss::Index *)index)->reconstruct_batch(n, keys, recons);
}

void index_reconstruct_n(Index *index, int64_t i0, int64_t ni,  float *recons)
{
    ((faiss::Index *)index)->reconstruct_n(i0, ni, recons);
}

void index_search_and_reconstruct(Index *index, int64_t n, const float *x, int64_t k, float *distances, int64_t *labels, float *recons, const SearchParameters *params)
{
    ((faiss::Index *)index)->search_and_reconstruct(n, x, k, distances, labels, recons, (faiss::SearchParameters *)params);
}

void index_compute_residual(const Index *index, const float *x, float *residual, const int64_t key)
{
    ((faiss::Index *)index)->compute_residual(x, residual, key);
}

void index_compute_residual_n(const Index *index, int64_t n, const float *x, float *residual, const int64_t *keys)
{
    ((faiss::Index *)index)->compute_residual_n(n, x, residual, keys);
}

DistanceComputer * index_get_distance_computer(const Index *index)
{
    return (DistanceComputer *)((faiss::Index *)index)->get_distance_computer();
}

size_t index_sa_code_size(const Index *index)
{
    return ((faiss::Index *)index)->sa_code_size();
}

void index_sa_encode(const Index *index, int64_t n, const float *x, uint8_t *bytes)
{
    ((faiss::Index *)index)->sa_encode(n, x, bytes);
}

void index_sa_decode(const Index *index, int64_t n, const uint8_t *bytes, float *x)
{
    ((faiss::Index *)index)->sa_decode(n, bytes, x);
}

void index_merge_from(Index *index, Index *rhs, int64_t add_id)
{
    ((faiss::Index *)index)->merge_from(*(faiss::Index *)rhs, add_id);
}

bool index_check_compatible_for_merge(const Index *index, const Index *rhs)
{
    try
    {
        ((faiss::Index *)index)->check_compatible_for_merge(*(faiss::Index *)rhs);
        return true;
    }
    catch (const faiss::FaissException& e)
    {
        return false;
    }
}

SearchParameters* search_parameters_new(IDSelector *sel)
{
    auto sp = new faiss::SearchParameters{};
    sp->sel = (faiss::IDSelector *)sel;
    return (SearchParameters *)sp;
}

void search_parameters_free(SearchParameters *sp)
{
    delete (faiss::SearchParameters *)sp;
}

IDSelector* search_parameters_sel(SearchParameters *sp)
{
    return (IDSelector *)((faiss::SearchParameters *)sp)->sel;
}

void search_parameters_set_sel(SearchParameters *sp, IDSelector *sel)
{
    ((faiss::SearchParameters *)sp)->sel = (faiss::IDSelector *)sel;
}