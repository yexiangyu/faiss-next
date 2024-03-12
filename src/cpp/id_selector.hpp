#include "types.hpp"

bool                id_selector_is_member(const IDSelector *sel, int64_t id);
void                id_selector_free(IDSelector *sel);
IDSelectorRange*    id_selector_range_new(int64_t imin, int64_t imax, bool assume_sorted);
int64_t             id_selector_range_imin(const IDSelectorRange *sel);
int64_t             id_selector_range_imax(const IDSelectorRange *sel);
bool                id_selector_range_assume_sorted(const IDSelectorRange *sel);
void                id_selector_range_find_sorted_ids_bounds(const IDSelectorRange *sel, size_t list_size, const int64_t *ids, size_t *jmin, size_t *jmax);
IDSelectorArray*    id_selector_array_new(int64_t n, const int64_t *ids);
const int64_t*      id_selector_array_ids(const IDSelectorArray *sel);
size_t              id_selector_array_n(const IDSelectorArray *sel);
IDSelectorBatch*    id_selector_batch_new(int64_t n, const int64_t *ids); // TODO: bloom filter for batch id selector
IDSelectorBitmap*   id_selector_bitmap_new(int64_t n, const uint8_t *ids);
IDSelectorNot*      id_selector_not_new(const IDSelector *sel);
IDSelectorAll*      id_selector_all_new();
IDSelectorAnd*      id_selector_and_new(const IDSelector *lhs, const IDSelector *rhs);
IDSelectorOr*       id_selector_or_new(const IDSelector *lhs, const IDSelector *rhs);
IDSelectorXOr*      id_selector_xor_new(const IDSelector *lhs, const IDSelector *rhs);