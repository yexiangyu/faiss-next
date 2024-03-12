#include "types.hpp"

void                        distance_computer_free(DistanceComputer *comp);
void                        distance_computer_set_query(DistanceComputer *comp, const float *);
float                       distance_computer_compute(DistanceComputer *comp, int64_t i); // operator()
float                       distance_computer_symmetric_dis(DistanceComputer *comp, int64_t i, int64_t j);

float                       flat_codes_distance_computer_distance_to_code(FlatCodesDistanceComputer *comp,  const uint8_t * code);