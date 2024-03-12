#include "distance_computer.hpp"
#include "faiss/impl/DistanceComputer.h"

void distance_computer_free(DistanceComputer *comp)
{
    delete (faiss::DistanceComputer *)(comp);
}

void distance_computer_set_query(DistanceComputer *comp, const float *x)
{
    ((faiss::DistanceComputer *)comp)->set_query(x);
}

float distance_computer_compute(DistanceComputer *comp, int64_t i)
{
    return ((faiss::DistanceComputer *)comp)->operator()(i);
}

float distance_computer_symmetric_dis(DistanceComputer *comp, int64_t i, int64_t j)
{
    return ((faiss::DistanceComputer *)comp)->symmetric_dis(i, j);
}

float flat_codes_distance_computer_distance_to_code(FlatCodesDistanceComputer *comp, const uint8_t *code)
{
    return ((faiss::FlatCodesDistanceComputer *)comp)->distance_to_code(code);
}