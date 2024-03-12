#include "types.hpp"

// IndexBinary*    index_binary_new(int64_t d, int32_t metric);
void            index_binary_free(IndexBinary *);
int32_t         index_binary_d(const IndexBinary *);
int32_t         index_binary_code_size(const IndexBinary *);
bool            index_binary_verbose(const IndexBinary *);
void            index_binary_set_verbose(IndexBinary *, bool verbose);
bool            index_binary_is_trained(const IndexBinary *);
int32_t         index_binary_metric_type(const IndexBinary *);
void            index_binary_train(IndexBinary *, int64_t n, const uint8_t *x);
void            index_binary_add(IndexBinary *, int64_t n, const uint8_t *x);        
void            index_binary_add_with_ids(IndexBinary *, int64_t n, const uint8_t *x, const int64_t *xids);
void            index_binary_search(const IndexBinary *, int64_t n, const uint8_t *x, int64_t k, int32_t *distances, int64_t *labels, const SearchParameters *params);
void            index_binary_range_search(const IndexBinary *, int64_t n, const uint8_t *x, int32_t radius, RangeSearchResult *result, const SearchParameters *params);
void            index_binary_assign(const IndexBinary *, int64_t n, const uint8_t *x, int64_t *labels, int64_t k);
size_t          index_binary_remove_ids(IndexBinary *,  const IDSelector *sel);
void            index_binary_reconstruct(const IndexBinary *, int64_t key, uint8_t *recons);
void            index_binary_reconstruct_n(const IndexBinary *, int64_t i0, int64_t ni, uint8_t *recons);
void            index_binary_search_and_reconstruct(const IndexBinary *, int64_t n, const uint8_t *x, int64_t k, int32_t *distances, int64_t *labels, uint8_t *recons, const SearchParameters *params);
void            index_binary_display(const IndexBinary *);
void            index_binary_merge_from(IndexBinary *, IndexBinary *, int64_t add_id);
bool            index_binary_check_compatible_for_merge(const IndexBinary *, const IndexBinary *);