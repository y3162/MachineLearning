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
        if (lossType == "MLE") {
            return new CG::MLE(output, target);
        } else {
            assert (false);
        }
    }



    NN::NN (CG::Leaf *input, CG::Leaf *target, CG::Node *output, CG::Node *loss)
    : input(input), target(target), output(output), loss(loss)
    {
        assert (loss->data.size() == 1);
    }

    vec1<dtype> NN::expect(const vec1<dtype> expectData)
    {
        input->getInput(expectData);

        input->forwardPropagation();
        target->forwardPropagation();

        return output->data;
    }

    dtype NN::test(const vec1<dtype> testData, const vec1<dtype> targetData)
    {
        input->getInput(testData);
        target->getInput(targetData);

        input->forwardPropagation();
        target->forwardPropagation();

        dtype ret = loss->data.at(0);
        return ret;
    }

    dtype NN::train(const vec1<dtype> trainData, const vec1<dtype> targetData)
    {
        input->getInput(trainData);
        target->getInput(targetData);

        input->forwardPropagation();
        target->forwardPropagation();
        loss->backwardPropagation();
        
        return loss->data.at(0);
    }

    void NN::update(dtype eta)
    {
        loss->update(eta);
    }



    NN* parseFeedForward(std::string filename)
    {
        CGP::Parser P;
        CG::Node *loss = P.parseAll(filename);

        CG::Node *output = loss->backward.at(0);
        CG::Leaf *target = dynamic_cast<CG::Leaf*>(loss->backward.at(1));

        CG::Node *temp = output;
        while (temp->backward.size() != 0) {
            assert (temp->backward.size() == 1);
            temp = temp->backward.at(0);
        }

        CG::Leaf *input = dynamic_cast<CG::Leaf*>(temp);

        return new NN(input, target, output, loss);
    }

    NN* feedForwardReLU(const vec1<size_t> nodes, std::string lossType)
    {   
        CG::Leaf* input  = new CG::Leaf(nodes.at(0));
        CG::Node* output = input;

        for (int i=0; i<nodes.size()-1; ++i) 
        {
            output = new CG::ReLU(output);
            output = new CG::Affine(output, initWeight("He", nodes.at(i), nodes.at(i + 1)));
        }

        CG::Leaf* target = new CG::Leaf(nodes.at(nodes.size() - 1));

        CG::Node* loss = setLossFunction(output, target, "MLE");

        return new NN(input, target, output, loss);
    }
}
