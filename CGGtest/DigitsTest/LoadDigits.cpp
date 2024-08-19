#include <fstream>
#include <iostream>
#include <cassert>
#include <sstream>
#include <vector>
#include "../../ComputationGraph/Type.hpp"
#include "LoadDigits.hpp"

vec2<dtype> loadDigitsData(std::string path)
{
    std::ifstream input(path);

    if (!input) {
        std::cerr << "File Cannot Open : " << path << std::endl;
        exit(1);
    }

    std::string buffer;
    std::string num;

    vec2<dtype> ret;
    ret.resize(1798);
    for (int i=0; i<1798; ++i) {
        ret.at(i).resize(8*8);
    }

    int i = 0, j = 0;

    while (std::getline(input, buffer)) {
        if (buffer.empty()) {
            break;
        }
        j = 0;
        std::stringstream ss(buffer);
        while (getline (ss, num, ',')) {
            if (num.empty()) {
                break;
            }
            ret.at(i).at(j++) = std::stod(num);
        }
        assert (j == 64);
        ++i;
    }
    assert (i == 1797);

    return ret;
}

vec2<dtype> loadDigitsTarget(std::string path)
{
    std::ifstream input(path);

    if (!input) {
        std::cerr << "File Cannot Open : " << path << std::endl;
        exit(1);
    }

    std::string buffer;
    std::string num;

    vec2<dtype> ret;
    ret.resize(1798);
    for (int i=0; i<1798; ++i) {
        ret.at(i).resize(10, 0);
    }

    int i = 0;

    std::getline(input, buffer);

    std::stringstream ss(buffer);

    while (getline (ss, num, ',')) {
        if (num.empty()) {
            break;
        }
        int temp = std::stoi(num);
        ret.at(i++).at(temp) = 1;
    }
    assert (i == 1797);

    return ret;
}
