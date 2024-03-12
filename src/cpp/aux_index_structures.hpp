#include "types.hpp"

RangeSearchResult*  range_search_result_new(size_t nq, bool alloc_lims);
void                range_search_result_free(RangeSearchResult *);
void                range_search_result_do_allocation(RangeSearchResult *);
size_t              range_search_result_nq(const RangeSearchResult *);
const size_t*       range_search_result_lims(const RangeSearchResult *);
const int64_t*      range_search_result_labels(const RangeSearchResult *);
const float*        range_search_result_distances(const RangeSearchResult *);
size_t              range_search_result_buffer_size(const RangeSearchResult *);