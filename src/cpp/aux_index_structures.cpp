#include "aux_index_structures.hpp"
#include "faiss/impl/AuxIndexStructures.h"

RangeSearchResult* range_search_result_new(size_t nq, bool alloc_lims) {
    return (RangeSearchResult *)(new faiss::RangeSearchResult(nq, alloc_lims));
}

void range_search_result_free(RangeSearchResult *rsr) {
    delete (faiss::RangeSearchResult *)rsr;
}

void range_search_result_do_allocation(RangeSearchResult *rsr) {
    ((faiss::RangeSearchResult *)rsr)->do_allocation();
}

size_t range_search_result_nq(const RangeSearchResult *rsr) {
    return ((faiss::RangeSearchResult *)rsr)->nq;
}

const size_t* range_search_result_lims(const RangeSearchResult *rsr) {
    return ((faiss::RangeSearchResult *)rsr)->lims;
}

const int64_t* range_search_result_labels(const RangeSearchResult *rsr) {
    return ((faiss::RangeSearchResult *)rsr)->labels;
}

const float* range_search_result_distances(const RangeSearchResult *rsr) {
    return ((faiss::RangeSearchResult *)rsr)->distances;
}

size_t range_search_result_buffer_size(const RangeSearchResult *rsr) {
    return ((faiss::RangeSearchResult *)rsr)->buffer_size;
}