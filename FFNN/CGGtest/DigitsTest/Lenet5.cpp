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
                data.at(i).at(j).at(k) = D.at(i).at(j * DIGITS_DATA_WIDTH + k);
            }
        }
    }

    //CGG::NN2d* cnn = CGG::Lenet5(DIGITS_DATA_HEIGHT, DIGITS_DATA_WIDTH);
    CGG::NN2d* cnn = CGG::parseLenet5("CEE.txt");

    int x = 0;
    int y = DIGITS_TRAIN_SIZE;

    for (int n=1; n<=10000; ++n) {
        double loss = 0;
        for (int i=0; i<100; ++i) {
            x = (x+1) % DIGITS_TRAIN_SIZE;
            loss += cnn->train(data.at(x), target.at(x));
        }

        int score = 0;
        for (int i=0; i<100; ++i) {
            y = (y + 1 - DIGITS_TRAIN_SIZE) % DIGITS_TEST_SIZE + DIGITS_TRAIN_SIZE;
            vec1<dtype> y_hat = cnn->expect(data.at(y));
            int expect = 0;
            dtype max = y_hat.at(0);
            for (int j=1; j<10; ++j) {
                if (max < y_hat.at(j)) {
                    expect =j;
                    max = y_hat.at(j);
                }
            }
            if (target.at(y).at(expect) == 1) {
                ++score;
            }
        }

        cnn->update(1e-3);
        std::cout << std::setw(3) << n << ": " << "train loss = " << std::setw(9) << std::fixed << std::setprecision(5) << loss << " accuracy = " << std::setw(7) << std::fixed << std::setprecision(5) << (double)score << "%" << std::endl;

        if (n%10==0) {
            CGC::Converter C;
            C.convertAll(cnn->loss, "CEE.txt");
            //C.convertAll(fnn->loss, "MSE.txt");
            std::cout << "--- This network was saved! ---" << std::endl;
        }
    }
}
