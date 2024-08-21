#ifndef CG_HPP
#define CG_HPP

#include <iostream>
#include <cassert>
#include <vector>
#include "Type.hpp"

namespace CG
{
    template<typename T> using vec1 = type::vec1<T>;
    template<typename T> using vec2 = type::vec2<T>;
    using dtype = type::dtype;

    class Node
    {
        public :
            size_t      domsize;
            size_t      height;
            size_t      width;
            vec1<dtype> data;
            vec1<dtype> grad;
            vec1<Node*> forward;
            vec1<Node*> backward;
            int         f_count = 0;
            int         b_count = 0;

            Node ();

            void pushThis(Node *node);

            virtual void calcData();
            void forwardPropagation();

            virtual void calcPartialDerivative();
            void backwardPropagation();

            virtual void updateParameters(dtype eta);
            void update(dtype eta);
    };

    class Leaf1 : public Node
    {
        public :
            Leaf1 (size_t size);

            void getInput(const vec1<dtype> input);
    };

    class Leaf2 : public Node
    {
        public :
            Leaf2 (size_t height, size_t width);

            void getInput(const vec1<dtype> input);
            void getInput(const vec2<dtype> input);
    };

    class Concatenation : public Node
    {
        public :
            vec1<size_t> dataSize;

            Concatenation (vec1<Node*> nodes);

            int whichNode(size_t index);

            virtual void calcData();

            virtual void calcPartialDerivative();
    };

    class MMtoM : public Node
    {
        public :
            MMtoM (Node *node1, Node *node2);
    };

    class MMto1 : public Node
    {
        public :
            MMto1 (Node *node1, Node *node2);
    };

    class MtoM : public Node
    {
        public :
            MtoM (Node *node1);
    };

    class Mto1 : public Node
    {
        public :
            Mto1 (Node *node1);
    };

    class Add : public MMtoM
    {   
        public :
            Add (Node *node1, Node *node2);

            virtual void calcData();
            virtual void calcPartialDerivative();
    };

    class Sub : public MMtoM
    {
        public :
            Sub (Node *node1, Node *node2);

            virtual void calcData();

            virtual void calcPartialDerivative();
    };

    class Dots : public MMto1
    {
        public :
            Dots (Node *node1, Node *node2);

            virtual void calcData();

            virtual void calcPartialDerivative();
    };

    class MSE : public MMto1
    {
        public : 
            MSE (Node *node1, Node *node2);

            virtual void calcData();

            virtual void calcPartialDerivative();
    };

    class CEE : public MMto1
    {
        public : 
            CEE (Node *node1, Node *node2);

            virtual void calcData();

            virtual void calcPartialDerivative();
    };

    class ReLU : public MtoM
    {
        public :
            ReLU (Node *node1);

            virtual void calcData();

            virtual void calcPartialDerivative();
    };

    class Softmax : public MtoM
    {
        public : 
            Softmax (Node *node1);

            virtual void calcData();

            virtual void calcPartialDerivative();
    };

    class Norm2 : public Mto1
    {
        public :
            Norm2 (Node *node1);

            virtual void calcData();
            
            virtual void calcPartialDerivative();
    };

    class Affine : public Node
    {
        public :
            vec2<dtype> weight;
            vec2<dtype> gradweight;
            const dtype       bias;
            
            Affine (Node *node1, const vec2<dtype> W, dtype b);
            Affine (Node *node1, const vec2<dtype> W);

            virtual void calcData();

            virtual void calcPartialDerivative();

            virtual void updateParameters(dtype eta);
    };

    class Convolution : public Node
    {
        public :
            vec2<dtype> kernel;
            vec2<dtype> gradKernel;
            dtype       bias;
            dtype       gradBias;
            size_t      pl, pt;
            size_t      sw;

            Convolution (Node *node1, const vec2<dtype> K, dtype b, size_t stride, size_t topPadding, size_t leftPadding, size_t height, size_t width);
            Convolution (Node *node1, const vec2<dtype> K, dtype b, size_t stride, size_t height, size_t width);
            Convolution (Node *node1, const vec2<dtype> K, dtype b, size_t stride);
            Convolution (Node *node1, const vec2<dtype> K, dtype b, size_t height, size_t width);
            Convolution (Node *node1, const vec2<dtype> K, dtype b);

            dtype getDomData(int col, int row);

            virtual void calcData();

            virtual void calcPartialDerivative();

            virtual void updateParameters(dtype eta);
    };

    void dumpNode (Node const node1, std::string name);
};

#endif
