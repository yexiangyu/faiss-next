#pragma once

using FaissIndexBinary = void;

int faiss_index_binary_factory(int d, const char *description, FaissIndexBinary **index);