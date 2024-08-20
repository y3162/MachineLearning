#include "CGgenerator.hpp"
#include "CGconverter.hpp"
#include "CGparser.hpp"
#include <string>
#include <random>

namespace CGG
{
    vec2<dtype> initWeight(std::string initType, size_t domSize, size_t ranSize)
    {
        vec2<dtype> ret;
        ret.resize(domSize + 1);
        for (int i=0; i<= domSize; ++i) {
            ret.at(i).resize(ranSize);
        }

        std::random_device seed_gen;
        std::default_random_engine engine(seed_gen());

        if (initType == "He") {
            std::normal_distribution<> dist(0, std::sqrt(2.0 / domSize));
            for (int i=0; i<=domSize; ++i) {
                for (int j=0; j<ranSize; ++j) {
                    ret.at(i).at(j) = dist(engine);
                }
            }
        } else if (initType == "Xavier") {
            std::normal_distribution<> dist(0, std::sqrt(2.0 / (domSize + ranSize)));
            for (int i=0; i<=domSize; ++i) {
                for (int j=0; j<ranSize; ++j) {
                    ret.at(i).at(j) = dist(engine);
                }
            }
        } else {
            assert (false);
        }

        return ret;
    }

    CG::Node* setLossFunction(CG::Node *output, CG::Node *target, std::string lossType)
    {
        if (lossType == "MSE") {
            return new CG::MSE(output, target);
        } else if (lossType == "CEE") {
            return new CG::CEE(output, target);
        } else {
            assert (false);
        }
    }

    CG::Node* setNormalizationFunction(CG::Node *output, std::string normalizationType)
    {
        if (normalizationType == "") {
            return output;
        } else if (normalizationType == "Softmax") {
            return new CG::Softmax(output);
        } else {
            assert (false);
        }
    }



    FNN::FNN (CG::Leaf1 *input, CG::Leaf1 *target, CG::Node *output, CG::Node *loss)
    : input(input), target(target), output(output), loss(loss)
    {
        assert (loss->data.size() == 1);
    }

    vec1<dtype> FNN::expect(const vec1<dtype> expectData)
    {
        input->getInput(expectData);

        input->forwardPropagation();
        target->forwardPropagation();

        return output->data;
    }

    dtype FNN::test(const vec1<dtype> testData, const vec1<dtype> targetData)
    {
        input->getInput(testData);
        target->getInput(targetData);

        input->forwardPropagation();
        target->forwardPropagation();

        return loss->data.at(0);
    }

    dtype FNN::train(const vec1<dtype> trainData, const vec1<dtype> targetData)
    {
        input->getInput(trainData);
        target->getInput(targetData);

        input->forwardPropagation();
        target->forwardPropagation();
        loss->backwardPropagation();
        
        return loss->data.at(0);
    }

    void FNN::update(dtype eta)
    {
        loss->update(eta);
    }



    FNN* parseFeedForward(std::string filename)
    {
        CGP::Parser P;
        CG::Node *loss = P.parseAll(filename);

        CG::Node  *output = loss->backward.at(0);
        CG::Leaf1 *target = dynamic_cast<CG::Leaf1*>(loss->backward.at(1));

        CG::Node *temp = output;
        while (temp->backward.size() != 0) {
            assert (temp->backward.size() == 1);
            temp = temp->backward.at(0);
        }

        CG::Leaf1 *input = dynamic_cast<CG::Leaf1*>(temp);

        return new FNN(input, target, output, loss);
    }

    FNN* feedForwardReLU(const vec1<size_t> nodes, std::string normlizationType, std::string lossType)
    {   
        CG::Leaf1* input = new CG::Leaf1(nodes.at(0));
        CG::Node* output = input;

        for (int i=0; i<nodes.size()-1; ++i) 
        {
            output = new CG::ReLU(output);
            output = new CG::Affine(output, initWeight("He", nodes.at(i), nodes.at(i + 1)));
        }
        
        output = setNormalizationFunction(output, normlizationType);

        CG::Leaf1* target = new CG::Leaf1(nodes.at(nodes.size() - 1));

        CG::Node* loss = setLossFunction(output, target, lossType);

        return new FNN(input, target, output, loss);
    }
}
