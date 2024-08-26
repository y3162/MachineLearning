#include <fstream>
#include <iostream>
#include <sstream>
#include "LoadDigits.hpp"
#include "../../ComputationGraph/CGconverter.hpp"
#include "../../ComputationGraph/CGgenerator.hpp"

int main(void) {

    vec2<dtype> data   = loadDigitsData("../../Data/Digits_data.csv");
    vec2<dtype> target = loadDigitsTarget("../../Data/Digits_target.csv");

    //CGG::NN1d* fnn = CGG::feedForwardReLU({784, 64, 64, 10}, "Softmax", "CEE");
    CGG::NN1d* fnn = CGG::parseFeedForward("CEE.txt");
    //CGG::NN1d* fnn = CGG::parseFeedForward("MSE.txt");
    //CGG::NN1d* fnn = CGG::feedForwardReLU({64, 64, 64, 10}, "Softmax", "MSE");

    for (int n=1; n<=1; ++n) {
        double loss = 0;
        for (int i=0; i<DIGITS_TRAIN_SIZE; ++i) {
            loss += fnn->train(data.at(i), target.at(i));
        }

        int score = 0;
        for (int i=0; i<DIGITS_TEST_SIZE; ++i) {
            vec1<dtype> y_hat = fnn->expect(data.at(i));
            int expect = 0;
            dtype max = y_hat.at(0);
            for (int j=1; j<10; ++j) {
                if (max < y_hat.at(j)) {
                    expect =j;
                    max = y_hat.at(j);
                }
            }
            if (target.at(i).at(expect) == 1) {
                ++score;
            }
        }

        fnn->update(0);
        std::cout << std::setw(3) << n << ": " << "train loss = " << std::setw(9) << std::fixed << std::setprecision(5) << loss << " accuracy = " << std::setw(7) << std::fixed << std::setprecision(5) << (double)score / 100 << "%" << std::endl;

        if (n%10==0) {
            CGC::Converter C;
            C.convertAll(fnn->loss, "CEE.txt");
            //C.convertAll(fnn->loss, "MSE.txt");
            std::cout << "--- This network was saved! ---" << std::endl;
        }
    }

}
