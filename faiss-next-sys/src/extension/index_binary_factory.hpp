#pragma once

using FaissIndexBinary = void;
using Char = void;

int faiss_index_binary_factory(int d, const Char *description, FaissIndexBinary **index);