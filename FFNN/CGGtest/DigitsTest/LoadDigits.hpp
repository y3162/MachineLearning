#ifndef LOADDIGITS_HPP
#define LOADDIGITS_HPP

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "../../ComputationGraph/Type.hpp"

#define DIGITS_DATA_SIZE 70000
#define DIGITS_DATA_HEIGHT 28
#define DIGITS_DATA_WIDTH 28
#define DIGITS_TRAIN_SIZE 60000
#define DIGITS_TEST_SIZE (DIGITS_DATA_SIZE-DIGITS_TRAIN_SIZE)

template<typename T> using vec1 = type::vec1<T>;
template<typename T> using vec2 = type::vec2<T>;
using dtype = type::dtype;

vec2<dtype> loadDigitsData(std::string path);
vec2<dtype> loadDigitsTarget(std::string path);

#endif