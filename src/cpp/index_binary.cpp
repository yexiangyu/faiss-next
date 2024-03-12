#include "index_binary.hpp"
#include "faiss/IndexBinary.h"

void index_binary_free(IndexBinary *index)
{
    delete (faiss::IndexBinary *)index;
}

int32_t index_binary_d(const IndexBinary *index)
{
    return ((faiss::IndexBinary *)index)->d;
}

int32_t index_binary_code_size(const IndexBinary *index)
{
    return ((faiss::IndexBinary *)index)->code_size;
}

bool index_binary_verbose(const IndexBinary *index)
{
    return ((faiss::IndexBinary *)index)->verbose;
}

void index_binary_set_verbose(IndexBinary *index, bool verbose)
{
    ((faiss::IndexBinary *)index)->verbose = verbose;
}

bool index_binary_is_trained(const IndexBinary *index)
{
    return ((faiss::IndexBinary *)index)->is_trained;
}

int32_t index_binary_metric_type(const IndexBinary *index)
{
    return ((faiss::IndexBinary *)index)->metric_type;
}

void index_binary_train(IndexBinary *index, int64_t n, const uint8_t *x)
{
    ((faiss::IndexBinary *)index)->train(n, x);
}

void index_binary_add(IndexBinary *index, int64_t n, const uint8_t *x)
{
    ((faiss::IndexBinary *)index)->add(n, x);
}

void index_binary_add_with_ids(IndexBinary *index, int64_t n, const uint8_t *x, const int64_t *xids)
{
    ((faiss::IndexBinary *)index)->add_with_ids(n, x, xids);
}

void index_binary_search(const IndexBinary *index, int64_t n, const uint8_t *x, int64_t k, int32_t *distances, int64_t *labels, const SearchParameters *params)
{
    ((faiss::IndexBinary *)index)->search(n, x, k, distances, labels, (faiss::SearchParameters*)params);
}

void index_binary_range_search(const IndexBinary *index, int64_t n, const uint8_t *x, int32_t radius, RangeSearchResult *result, const SearchParameters *params)
{
    ((faiss::IndexBinary *)index)->range_search(n, x, radius, (faiss::RangeSearchResult*)result, (faiss::SearchParameters*)params);
}

void index_binary_assign(const IndexBinary *index, int64_t n, const uint8_t *x, int64_t *labels, int64_t k)
{
    ((faiss::IndexBinary *)index)->assign(n, x, labels, k);
}

size_t index_binary_remove_ids(IndexBinary *index,  const IDSelector *sel)
{
    return ((faiss::IndexBinary *)index)->remove_ids(*(faiss::IDSelector *)sel);
}

void index_binary_reconstruct(const IndexBinary *index, int64_t key, uint8_t *recons)
{
    ((faiss::IndexBinary *)index)->reconstruct(key, recons);
}

void index_binary_reconstruct_n(const IndexBinary *index, int64_t i0, int64_t ni, uint8_t *recons)
{
    ((faiss::IndexBinary *)index)->reconstruct_n(i0, ni, recons);
}

void index_binary_search_and_reconstruct(const IndexBinary *index, int64_t n, const uint8_t *x, int64_t k, int32_t *distances, int64_t *labels, uint8_t *recons, const SearchParameters *params)
{
    ((faiss::IndexBinary *)index)->search_and_reconstruct(n, x, k, distances, labels, recons, (faiss::SearchParameters*)params);
}

void index_binary_display(const IndexBinary *index)
{
    ((faiss::IndexBinary *)index)->display();
}

void index_binary_merge_from(IndexBinary *index, IndexBinary *other, int64_t add_id)
{
    ((faiss::IndexBinary *)index)->merge_from(*(faiss::IndexBinary *)other, add_id);
}

bool index_binary_check_compatible_for_merge(const IndexBinary *index, const IndexBinary *other)
{
    try {
        ((faiss::IndexBinary *)index)->check_compatible_for_merge(*(faiss::IndexBinary *)other);
        return true;
    } 
    catch(const faiss::FaissException& e) {
        return false;
    }
}