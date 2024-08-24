#ifndef CGG_H
#define CGG_H

#include "CG.hpp"
#include <string>

namespace CGG
{
    template<typename T> using vec1 = CG::vec1<T>;
    template<typename T> using vec2 = CG::vec2<T>;
    template<typename T> using vec3 = type::vec3<T>;
    using dtype = CG::dtype;

    vec2<dtype> initWeight(std::string initType, size_t domSize, size_t ranSize);
    vec3<dtype> initKernel(std::string initType, size_t channel, size_t height, size_t width);

    CG::Node* setLossFunction(CG::Node *output, CG::Node *target, std::string lossType);

    CG::Node* setNormalizationFunction(CG::Node *output, std::string normalizationType);

    class NN1d
    {
        public : 
            CG::Leaf1 *input;
            CG::Leaf1 *target;
            CG::Node  *output;
            CG::Node  *loss;

            NN1d (CG::Leaf1 *input, CG::Leaf1 *target, CG::Node *output, CG::Node *loss);

            vec1<dtype> expect(vec1<dtype> expectData);

            dtype test(vec1<dtype> testData, vec1<dtype> targetData);

            dtype train(vec1<dtype> trainData, vec1<dtype> targetData);

            void update(dtype eta);
    };
    class NN2d
    {
        public : 
            CG::Leaf2 *input;
            CG::Leaf1 *target;
            CG::Node  *output;
            CG::Node  *loss;

            NN2d (CG::Leaf2 *input, CG::Leaf1 *target, CG::Node *output, CG::Node *loss);

            vec1<dtype> expect(vec2<dtype> expectData);

            dtype test(vec2<dtype> testData, vec1<dtype> targetData);

            dtype train(vec2<dtype> trainData, vec1<dtype> targetData);

            void update(dtype eta);
    };

    NN1d* parseFeedForward(std::string filename);

    NN1d* feedForwardReLU(vec1<size_t> nodes, std::string normalizationType, std::string lossType);

    NN2d* Lenet5(size_t height, size_t width);
}

#endif
