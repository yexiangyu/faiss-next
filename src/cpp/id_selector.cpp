#include "faiss/impl/IDSelector.h"
#include "id_selector.hpp"


bool id_selector_is_member(const IDSelector *sel, int64_t id)
{
    return ((faiss::IDSelector*)sel)->is_member(id);
}

void id_selector_free(IDSelector *sel)
{
    delete (faiss::IDSelector*)sel;
}

IDSelectorRange *id_selector_range_new(int64_t imin, int64_t imax, bool assume_sorted)
{
    return (IDSelectorRange*)new faiss::IDSelectorRange(imin, imax, assume_sorted);
}

int64_t id_selector_range_imin(const IDSelectorRange *sel)
{
    return ((faiss::IDSelectorRange*)sel)->imin;
}

int64_t id_selector_range_imax(const IDSelectorRange *sel)
{
    return ((faiss::IDSelectorRange*)sel)->imax;
}

bool id_selector_range_assume_sorted(const IDSelectorRange *sel)
{
    return ((faiss::IDSelectorRange*)sel)->assume_sorted;
}

void id_selector_range_set_assume_sorted(IDSelectorRange *sel, bool assume_sorted)
{
    ((faiss::IDSelectorRange*)sel)->assume_sorted = assume_sorted;
}

IDSelectorArray *id_selector_array_new(int64_t n, const int64_t *ids)
{
    return (IDSelectorArray*)new faiss::IDSelectorArray(n, ids);
}

const int64_t *id_selector_array_ids(const IDSelectorArray *sel)
{
    return ((faiss::IDSelectorArray*)sel)->ids;
}

size_t id_selector_array_n(const IDSelectorArray *sel)
{
    return ((faiss::IDSelectorArray*)sel)->n;
}

IDSelectorBatch *id_selector_batch_new(int64_t n, const int64_t *ids)
{
    return (IDSelectorBatch*)new faiss::IDSelectorBatch(n, ids);
}


IDSelectorBitmap *id_selector_bitmap_new(int64_t n, const uint8_t *ids)
{
    return (IDSelectorBitmap*)new faiss::IDSelectorBitmap(n, ids);
}

IDSelectorNot *id_selector_not_new(const IDSelector *sel)
{
    return (IDSelectorNot*)new faiss::IDSelectorNot((faiss::IDSelector*)sel);
}

IDSelectorAll *id_selector_all_new()
{
    return (IDSelectorAll*)new faiss::IDSelectorAll();
}

IDSelectorAnd *id_selector_and_new(const IDSelector *lhs, const IDSelector *rhs)
{
    return (IDSelectorAnd*)new faiss::IDSelectorAnd((faiss::IDSelector*)lhs, (faiss::IDSelector*)rhs);
}

IDSelectorOr *id_selector_or_new(const IDSelector *lhs, const IDSelector *rhs)
{
    return (IDSelectorOr*)new faiss::IDSelectorOr((faiss::IDSelector*)lhs, (faiss::IDSelector*)rhs);
}

IDSelectorXOr *id_selector_xor_new(const IDSelector *lhs, const IDSelector *rhs)
{
    return (IDSelectorXOr*)new faiss::IDSelectorXOr((faiss::IDSelector*)lhs, (faiss::IDSelector*)rhs);
}
