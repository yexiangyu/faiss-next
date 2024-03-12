#include "index_factory.hpp"
#include "faiss/index_factory.h"

Index *index_factory(int d, const char *description, int metric)
{
	return (int *)faiss::index_factory(d, description, faiss::MetricType(metric));
}

IndexBinary* index_binary_factory(int d, const char *description)
{
    return (int *)faiss::index_binary_factory(d, description);
}