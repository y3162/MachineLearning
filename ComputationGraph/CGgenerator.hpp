#ifndef CGG_H
#define CGG_H

#include "CG.hpp"
#include <string>

namespace CGG
{
    template<typename T> using vec1 = CG::vec1<T>;
    template<typename T> using vec2 = CG::vec2<T>;
    using dtype = CG::dtype;

    vec2<dtype> initWeight(std::string initType, size_t domSize, size_t ranSize);

    CG::Node* setLossFunction(CG::Node *output, CG::Node *target, std::string lossType);

    class NN
    {
        public :
            CG::Leaf *input;
            CG::Leaf *target;
            CG::Node *output;
            CG::Node *loss;

            NN (CG::Leaf *input, CG::Leaf *target, CG::Node *output, CG::Node *loss);

            vec1<dtype> expect(const vec1<dtype> expectData);

            dtype test(const vec1<dtype> testData, const vec1<dtype> targetData);

            dtype train(const vec1<dtype> trainData, const vec1<dtype> targetData);

            void update(dtype eta);
    };

    NN* parseFeedForward(std::string filename);

    NN* feedForwardReLU(const vec1<size_t> nodes, std::string lossType);
}

#endif
