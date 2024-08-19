#ifndef LOADDIGITS_HPP
#define LOADDIGITS_HPP

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "../../ComputationGraph/Type.hpp"

template<typename T> using vec1 = type::vec1<T>;
template<typename T> using vec2 = type::vec2<T>;
using dtype = type::dtype;

vec2<dtype> loadDigitsData(std::string path);
vec2<dtype> loadDigitsTarget(std::string path);

#endif