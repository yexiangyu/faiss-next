#include <exception>
#include "faiss/index_factory.h"
#include "index_binary_factory.hpp"

int faiss_index_binary_factory(int d, const Char *description, FaissIndexBinary **index)
{
    try
    {
        auto i = faiss::index_binary_factory(d, (const char *)description);
        if (i)
        {
            *index = (FaissIndexBinary *)i;
            return 0;
        }
        return 999999999;
    }
    catch (std::exception e)
    {
        return 999999999;
    }
}