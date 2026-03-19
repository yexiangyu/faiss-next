#include <exception>
#include <faiss/index_factory.h>
#include "index_binary_factory.hpp"

int faiss_index_binary_factory(int d, const char *description, FaissIndexBinary **index)
{
    try
    {
        auto i = faiss::index_binary_factory(d, description);
        if (i)
        {
            *index = (FaissIndexBinary *)i;
            return 0;
        }
        return -1;
    }
    catch (std::exception &e)
    {
        return -1;
    }
}