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

    vec3<dtype> initKernel(std::string initType, size_t channel, size_t height, size_t width)
    {
        vec3<dtype> ret;
        ret.resize(channel);
        for (int i=0; i<channel; ++i) {
            ret.at(i) = initWeight(initType, height-1, width);
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



    NN1d::NN1d (CG::Leaf1 *input, CG::Leaf1 *target, CG::Node *output, CG::Node *loss)
    : input(input), target(target), output(output), loss(loss)
    {
        assert (loss->data.size() == 1);
    }

    vec1<dtype> NN1d::expect(vec1<dtype> expectData)
    {
        input->getInput(expectData);
        input->forwardPropagation();

        target->forwardPropagation();

        return output->data;
    }

    dtype NN1d::test(vec1<dtype> testData, vec1<dtype> targetData)
    {
        input->getInput(testData);
        target->getInput(targetData);

        input->forwardPropagation();
        target->forwardPropagation();

        return loss->data.at(0);
    }

    dtype NN1d::train(vec1<dtype> trainData, vec1<dtype> targetData)
    {
        input->getInput(trainData);
        target->getInput(targetData);

        input->forwardPropagation();
        target->forwardPropagation();
        loss->backwardPropagation();
        
        return loss->data.at(0);
    }

    void NN1d::update(dtype eta)
    {
        loss->update(eta);
    }



    NN2d::NN2d (CG::Leaf2 *input, CG::Leaf1 *target, CG::Node *output, CG::Node *loss)
    : input(input), target(target), output(output), loss(loss)
    {
        assert (loss->data.size() == 1);
    }

    vec1<dtype> NN2d::expect(vec2<dtype> expectData)
    {
        input->getInput(expectData);
        input->forwardPropagation();

        target->forwardPropagation();

        return output->data;
    }

    dtype NN2d::test(vec2<dtype> testData, vec1<dtype> targetData)
    {
        input->getInput(testData);
        target->getInput(targetData);

        input->forwardPropagation();
        target->forwardPropagation();

        return loss->data.at(0);
    }

    dtype NN2d::train(vec2<dtype> trainData, vec1<dtype> targetData)
    {
        input->getInput(trainData);
        target->getInput(targetData);

        input->forwardPropagation();
        target->forwardPropagation();
        loss->backwardPropagation();
        
        return loss->data.at(0);
    }

    void NN2d::update(dtype eta)
    {
        loss->update(eta);
    }



    NN1d* parseFeedForward(std::string filename)
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

        return new NN1d(input, target, output, loss);
    }

    NN2d* parseLenet5(std::string filename)
    {
        CGP::Parser P;
        CG::Node *loss = P.parseAll(filename);

        CG::Node  *output = loss->backward.at(0);
        CG::Leaf1 *target = dynamic_cast<CG::Leaf1*>(loss->backward.at(1));

        CG::Node *temp = output;
        while (temp->backward.size() != 0) {
            temp = temp->backward.at(0);
        }

        CG::Leaf2 *input = dynamic_cast<CG::Leaf2*>(temp);

        return new NN2d(input, target, output, loss);
    }

    NN1d* feedForwardReLU(vec1<size_t> nodes, std::string normlizationType, std::string lossType)
    {   
        CG::Leaf1* input = new CG::Leaf1(nodes.at(0));
        CG::Node* output = input;

        for (int i=0; i<nodes.size()-1; ++i) 
        {
            output = new CG::ReLU(output);
            output = new CG::Affine(output, initWeight("He", nodes.at(i), nodes.at(i + 1)), 1);
        }
        
        output = setNormalizationFunction(output, normlizationType);

        CG::Leaf1* target = new CG::Leaf1(nodes.at(nodes.size() - 1));

        CG::Node* loss = setLossFunction(output, target, lossType);

        return new NN1d(input, target, output, loss);
    }

    NN2d* Lenet5(size_t height, size_t width)
    {
        /* Input Layer */
        CG::Leaf2 *i0 = new CG::Leaf2(height, width);



        /* C1- Convolution Layer */
        vec1<CG::Convolution2d*> c1 = vec1<CG::Convolution2d*>(6);
        for (int i=0; i<6; ++i) {
            c1.at(i) = new CG::Convolution2d({i0}, initKernel("He", 1, 5, 5), 0, 28, 28);
        }

        vec1<CG::ReLU*> a1 = vec1<CG::ReLU*>(6);
        for (int i=0; i<6; ++i) {
            a1.at(i) = new CG::ReLU(c1.at(i));
        }


        
        /* S2- Pooling Layer */
        vec1<CG::AveragePooling2d*> s2 = vec1<CG::AveragePooling2d*>(6);
        for (int i=0; i<6; ++i) {
            s2.at(i) = new CG::AveragePooling2d(a1.at(i), 2, 2, 2);
        }

        vec1<CG::ReLU*> a2 = vec1<CG::ReLU*>(6);
        for (int i=0; i<6; ++i) {
            a2.at(i) = new CG::ReLU(s2.at(i));
        }



        /* C3- Convolution Layer */
        vec1<CG::Convolution2d*> c3 = {  new CG::Convolution2d({a2.at(0), a2.at(1), a2.at(2)}, initKernel("He", 3, 5, 5), 0)
                                       , new CG::Convolution2d({a2.at(1), a2.at(2), a2.at(3)}, initKernel("He", 3, 5, 5), 0)
                                       , new CG::Convolution2d({a2.at(2), a2.at(3), a2.at(4)}, initKernel("He", 3, 5, 5), 0)
                                       , new CG::Convolution2d({a2.at(3), a2.at(4), a2.at(5)}, initKernel("He", 3, 5, 5), 0)
                                       , new CG::Convolution2d({a2.at(4), a2.at(5), a2.at(0)}, initKernel("He", 3, 5, 5), 0)
                                       , new CG::Convolution2d({a2.at(5), a2.at(0), a2.at(1)}, initKernel("He", 3, 5, 5), 0)
                                       , new CG::Convolution2d({a2.at(0), a2.at(1), a2.at(2), a2.at(3)}, initKernel("He", 4, 5, 5), 0)
                                       , new CG::Convolution2d({a2.at(1), a2.at(2), a2.at(3), a2.at(4)}, initKernel("He", 4, 5, 5), 0)
                                       , new CG::Convolution2d({a2.at(2), a2.at(3), a2.at(4), a2.at(5)}, initKernel("He", 4, 5, 5), 0)
                                       , new CG::Convolution2d({a2.at(3), a2.at(4), a2.at(5), a2.at(0)}, initKernel("He", 4, 5, 5), 0)
                                       , new CG::Convolution2d({a2.at(4), a2.at(5), a2.at(0), a2.at(1)}, initKernel("He", 4, 5, 5), 0)
                                       , new CG::Convolution2d({a2.at(5), a2.at(0), a2.at(1), a2.at(2)}, initKernel("He", 4, 5, 5), 0)
                                       , new CG::Convolution2d({a2.at(0), a2.at(1), a2.at(3), a2.at(4)}, initKernel("He", 4, 5, 5), 0)
                                       , new CG::Convolution2d({a2.at(1), a2.at(2), a2.at(4), a2.at(5)}, initKernel("He", 4, 5, 5), 0)
                                       , new CG::Convolution2d({a2.at(2), a2.at(3), a2.at(5), a2.at(0)}, initKernel("He", 4, 5, 5), 0)
                                       , new CG::Convolution2d({a2.at(0), a2.at(1), a2.at(2), a2.at(3), a2.at(4), a2.at(5)}, initKernel("He", 6, 5, 5), 0)};

        vec1<CG::ReLU*> a3 = vec1<CG::ReLU*>(16);
        for (int i=0; i<16; ++i) {
            a3.at(i) = new CG::ReLU(c3.at(i));
        }



        /* S4- Pooling Layer */
        vec1<CG::AveragePooling2d*> s4 = vec1<CG::AveragePooling2d*>(16);
        for (int i=0; i<16; ++i) {
            s4.at(i) = new CG::AveragePooling2d(a3.at(i), 2, 2, 2);
        }

        vec1<CG::ReLU*> a4 = vec1<CG::ReLU*>(16);
        for (int i=0; i<16; ++i) {
            a4.at(i) = new CG::ReLU(s4.at(i));
        }



        /* C5- Convolution Layer */
        vec1<CG::Node*> arg5 = vec1<CG::Node*>(16);
        for (int i=0; i<16; ++i) {
            arg5.at(i) = a4.at(i);
        }
        vec1<CG::Convolution2d*> c5 = vec1<CG::Convolution2d*>(120);
        for (int i=0; i<120; ++i) {
            c5.at(i) = new CG::Convolution2d(arg5, initKernel("He", 16, 5, 5), 0);
        }

        vec1<CG::ReLU*> a5 = vec1<CG::ReLU*>(120);
        for (int i=0; i<120; ++i) {
            a5.at(i) = new CG::ReLU(c5.at(i));
        }



        /* F6- Fully Connected Layer */
        vec1<CG::Node*> arg6 = vec1<CG::Node*>(120);
        for (int i=0; i<120; ++i) {
            arg6.at(i) = a5.at(i);
        }
        CG::Node* c6 = new CG::Concatenation(arg6);
        CG::Affine* f6 = new CG::Affine(c6, initWeight("He", 120, 10), 1);
        CG::Softmax* o6 = new CG::Softmax(f6);



        /* Target Layer */
        CG::Leaf1* target = new CG::Leaf1(10);



        /* Output Layer */
        CG::CEE* l7 = new CG::CEE(o6, target);

        

        return new NN2d(i0, target, o6, l7);
    }
}
