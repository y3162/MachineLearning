#include <fstream>
#include <iostream>
#include <sstream>
#include "LoadDigits.hpp"
#include "../../ComputationGraph/CGconverter.hpp"
#include "../../ComputationGraph/CGgenerator.hpp"

int main(void) {

    vec2<dtype> D      = loadDigitsData("../../Data/Digits_data.csv");
    vec2<dtype> target = loadDigitsTarget("../../Data/Digits_target.csv");

    CG::vec3<dtype> data;
    data.resize(DIGITS_DATA_SIZE);
    for (int i=0; i<D.size(); ++i) {
        data.at(i).resize(DIGITS_DATA_HEIGHT);
        for (int j=0; j<DIGITS_DATA_HEIGHT; ++j) {
            data.at(i).at(j).resize(DIGITS_DATA_WIDTH);
            for (int k=0; k<DIGITS_DATA_WIDTH; ++k) {
                data.at(i).at(j).at(k) = D.at(i).at(j * DIGITS_DATA_W