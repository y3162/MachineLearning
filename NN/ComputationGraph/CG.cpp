#include <iostream>
#include <cassert>
#include <vector>
#include "CG.hpp"
#include "Type.hpp"

namespace CG
{   
    Node::Node (size_t domsize, size_t height, size_t width)
    : domsize(domsize), height(height), width(width), dsize(height * width)
    {
        forward.resize(0);

        data.resize(1);
        data.at(0).resize(dsize);
        grad.resize(1);
        grad.at(0).resize(dsize);
        f_count.resize(1);
        b_count.resize(1);
    }

    void Node::pushThis(Node *node) // push this as argument's forward node
    {
        size_t fsize = node->forward.size();
        node->forward.push_back(this);
    }

    void Node::calcData(){}
    void Node::forwardPropagation(ttype time)
    {
        size_t T = data.size();
        assert (T == grad.size());
        if (T <= time) {
            data.resize(time + 1);
            grad.resize(time + 1);
            for (int t=T; t<=time; ++t) {
                data.at(t).resize(dsize);
                grad.at(t).resize(dsize);
            }
            f_count.resize(time + 1);
            b_count.resize(time + 1);
        }

        if (++b_count.at(time) < backward.size()) {
            return;
        } else { // When all the forward passes from the units in the preceding layer have been completed
            b_count.at(time) = 0;
        }
        
        for (int t=0; t<data.size(); ++t) {
            for (int i=0; i<dsize; ++i) {
                grad.at(t).at(i) = 0;
            }
        }

        this->time = time;
        calcData();

        for (int i=0; i<forward.size(); ++i) {
            forward.at(i)->forwardPropagation(time);
        }
    }
    void Node::forwardPropagation()
    {
        forwardPropagation(0);
    }

    void Node::calcPartialDerivative(){}
    void Node::backwardPropagation(ttype time)
    {   
        if (++f_count.at(time) < forward.size()) {
            return;
        } else { // When all the backpropagations from the units in the next layer have been completed
            f_count.at(time) = 0;
        }

        if (forward.size() == 0) {
            assert (dsize == 1);
            grad.at(time).at(0) = 1;
        }

        this->time = time;
        calcPartialDerivative();

        for (int i=0; i<backward.size(); ++i) {
            backward.at(i)->backwardPropagation(time);
        }
    }
    void Node::backwardPropagation()
    {
        backwardPropagation(0);
    }

    void Node::updateParameters(dtype eta){}
    void Node::update(dtype eta, ttype time)
    {
        if (++f_count.at(time) < forward.size()) {
            return;
        } else { // When all the backpropagations from the units in the next layer have been completed
            f_count.at(time) = 0;
        }

        this->time = time;
        updateParameters(eta);

        for (int i=0; i<backward.size(); ++i) {
            backward.at(i)->update(eta, time);
        }
    }
    void Node::update(dtype eta)
    {
        update(eta, 0);
    }



    Leaf1::Leaf1 (size_t size)
    : Node (0, size, 1)
    {
        backward.resize(0);
    }

    void Leaf1::getInput(vec1<dtype> input, ttype time)
    {
        assert (dsize == input.size());
        size_t T = data.size();
        if (time >= T) {
            data.resize(time + 1);
            grad.resize(time + 1);
            for (int t=T; t<=time; ++t) {
                data.at(t).resize(dsize);
                grad.at(t).resize(dsize);
            }
            f_count.resize(time + 1);
            b_count.resize(time + 1);
        }

        //data.at(time) = input;
        for (int i=0; i<dsize; ++i) {
            data.at(time).at(i) = input.at(i);
        }
    }
    void Leaf1::getInput(vec1<dtype> input)
    {
        getInput(input, 0);
    }



    Leaf2::Leaf2 (size_t height, size_t width)
    : Node (0, height, width)
    {
        backward.resize(0);
    }

    void Leaf2::getInput(vec1<dtype> input, ttype time)
    {
        assert (dsize == input.size());
        int T = data.size();
        if (time >= T) {
            data.resize(time + 1);
            grad.resize(time + 1);
            for (int t=T-1; t<=time; ++t) {
                data.at(t).resize(dsize);
                grad.at(t).resize(dsize);
            }
            f_count.resize(time + 1);
            b_count.resize(time + 1);
        }

        //data.at(time) = input;
        for (int i=0; i<dsize; ++i) {
            data.at(time).at(i) = input.at(i);
        }
    }
    void Leaf2::getInput(vec1<dtype> input)
    {
        getInput(input, 0);
    }

    void Leaf2::getInput(vec2<dtype> input, ttype time)
    {
        assert (height == input.size());
        int T = data.size();
        if (time >= T) {
            data.resize(time + 1);
            grad.resize(time + 1);
            for (int t=T-1; t<=time; ++t) {
                data.at(t).resize(dsize);
                grad.at(t).resize(dsize);
            }
            f_count.resize(time + 1);
            b_count.resize(time + 1);
        }

        for (int i=0; i<height; ++i) {
            assert (input.at(i).size() == width);
            for (int j=0; j<width; ++j) {
                data.at(time).at(i * width + j) = input.at(i).at(j);
            }
        }
    }
    void Leaf2::getInput(vec2<dtype> input)
    {
        getInput(input, 0);
    }



    Concatenation::Concatenation (vec1<Node*> nodes)
    : Node (getSumSizeOfData(nodes), getSumSizeOfHeight(nodes), 1)
    {
        dataSize.resize(nodes.size());
        for (int i=0; i<nodes.size(); ++i) {
            assert (nodes.at(i)->width == 1);
            assert (nodes.at(i)->dsize == nodes.at(i)->height);
            if (i>0) {
                dataSize.at(i) = dataSize.at(i-1) + nodes.at(i-1)->dsize;
            }
        }

        backward.resize(nodes.size());
        for (int i=0; i<nodes.size(); ++i) {
            backward.at(i) = nodes.at(i);
            pushThis(nodes.at(i));
        }
    }

    int Concatenation::whichNode(size_t index)
    {
        assert (index < domsize);
        for (int i=0; i<backward.size()-1; ++i) {
            if (dataSize.at(i) <= index && index < dataSize.at(i+1)) {
                return i;
            }
        }
        return backward.size() - 1;
    }

    void Concatenation::calcData()
    {
        for (int i=0; i<domsize; ++i) {
            int index = whichNode(i);
            size_t remain = i - dataSize.at(index);
            data.at(time).at(i) = backward.at(index)->data.at(time).at(remain);
        }
    }

    void Concatenation::calcPartialDerivative()
    {
        for (int i=0; i<domsize; ++i) {
            int index = whichNode(i);
            size_t remain = i - dataSize.at(index);
            backward.at(index)->grad.at(time).at(remain) = grad.at(time).at(i);
        }
    }



    MMtoM::MMtoM (Node *node1, Node *node2)
    : Node (node1->dsize, node1->height, node1->width)
    {   
        assert (node1->dsize == node2->dsize);
        assert (node1->height == node2->height);
        assert (node1->width  == node2->width);

        backward.resize(2);
        backward.at(0) = node1;
        backward.at(1) = node2;

        pushThis(node1);
        pushThis(node2);
    };

    
    
    MMto1::MMto1 (Node *node1, Node *node2)
    : Node (node1->dsize, 1, 1)
    {   
        assert (node1->dsize == node2->dsize);
        assert (node1->height == node2->height);
        assert (node1->width  == node2->width);

        backward.resize(2);
        backward.at(0) = node1;
        backward.at(1) = node2;

        pushThis(node1);
        pushThis(node2);
    };



    MtoM::MtoM (Node *node1)
    : Node (node1->dsize, node1->height, node1->width)
    {
        backward.resize(1);
        backward.at(0) = node1;

        pushThis(node1);
    };



    Mto1::Mto1 (Node *node1)
    : Node (node1->dsize, 1, 1)
    {
        backward.resize(1);
        backward.at(0) = node1;

        pushThis(node1);
    };



    Filter2d::Filter2d (vec1<Node*> nodes, size_t kernelHeight, size_t kernelWidth, size_t stride, size_t topPadding, size_t leftPadding, size_t height, size_t width)
    : Node (nodes.at(0)->dsize, height, width), kheight(kernelHeight), kwidth(kernelWidth), pt(topPadding), pl(leftPadding), sw(stride)
    {
        size_t channel = nodes.size();

        for (int c=0; c<channel; ++c) {
            assert (kernelHeight <= nodes.at(c)->height);
            assert (kernelWidth  <= nodes.at(c)->width);
            assert (nodes.at(c)->height * nodes.at(c)->width == nodes.at(c)->dsize);
        }

        backward.resize(channel);
        for (int c=0; c<channel; ++c) {
            backward.at(c) = nodes.at(c);
            pushThis(nodes.at(c));
        }
    }

    bool Filter2d::inDomain(int col, int row)
    {
        size_t bheight = backward.at(0)->height;
        size_t bwidth  = backward.at(0)->width;
        if (0 <= col && col < bheight && 0 <= row && row < bwidth) {
            return true;
        } else {
            return false;
        }
    }

    dtype Filter2d::getDomData(int index, int col, int row)
    {
        size_t bwidth  = backward.at(index)->width;
        if (inDomain(col, row)) {
            return backward.at(index)->data.at(time).at(col * bwidth + row);
        } else {
            return 0;
        }
    }

    dtype Filter2d::getDomData(int col, int row)
    {
        return getDomData(0, col, row);
    }



    Add::Add (Node *node1, Node *node2) : MMtoM (node1, node2){};

    void Add::calcData()
    {
        for (int i=0; i<domsize; ++i) {
            data.at(time).at(i) = backward.at(0)->data.at(time).at(i) + backward.at(1)->data.at(time).at(i);
        }
    }

    void Add::calcPartialDerivative()
    {   
        for (int i=0; i<domsize; ++i) {
            backward.at(0)->grad.at(time).at(i) += 1 * grad.at(time).at(i);
            backward.at(1)->grad.at(time).at(i) += 1 * grad.at(time).at(i);
        }
    }



    Sub::Sub (Node *node1, Node *node2) : MMtoM (node1, node2){};

    void Sub::calcData()
    {
        for (int i=0; i<domsize; ++i) {
            data.at(time).at(i) = backward.at(0)->data.at(time).at(i) - backward.at(1)->data.at(time).at(i);
        }
    }

    void Sub::calcPartialDerivative()
    {
        for (int i=0; i<domsize; ++i) {
            backward.at(0)->grad.at(time).at(i) +=  1 * grad.at(time).at(i);
            backward.at(1)->grad.at(time).at(i) += -1 * grad.at(time).at(i);
        }
    }



    Dots::Dots (Node *node1, Node *node2) : MMto1 (node1, node2){};

    void Dots::calcData()
    {   
        data.at(time).at(0) = 0;
        for (int i=0; i<domsize; ++i) {
            data.at(time).at(0) += backward.at(0)->data.at(time).at(i) * backward.at(1)->data.at(time).at(i);
        }
    }

    void Dots::calcPartialDerivative()
    {
        for (int i=0; i<domsize; ++i) {
            backward.at(0)->grad.at(time).at(i) += backward.at(1)->data.at(time).at(i) * grad.at(time).at(0);
            backward.at(1)->grad.at(time).at(i) += backward.at(0)->data.at(time).at(i) * grad.at(time).at(0);
        }
    }



    MSE::MSE (Node *node1, Node *node2) : MMto1 (node1, node2){};

    void MSE::calcData()
    {   
        data.at(time).at(0) = 0;
        for (int i=0; i<domsize; ++i) {
            dtype err = backward.at(0)->data.at(time).at(i) - backward.at(1)->data.at(time).at(i);
            data.at(time).at(0) += err * err;
        }
        data.at(time).at(0) /= domsize;
    }

    void MSE::calcPartialDerivative()
    {
        for (int i=0; i<domsize; ++i) {
            dtype err = backward.at(0)->data.at(time).at(i) - backward.at(1)->data.at(time).at(i);
            backward.at(0)->grad.at(time).at(i) +=   2 * err * grad.at(time).at(0) / domsize;
            backward.at(1)->grad.at(time).at(i) += - 2 * err * grad.at(time).at(0) / domsize;
        }
    }



    CEE::CEE (Node *node1, Node *node2) : MMto1 (node1, node2){};

    void CEE::calcData()
    {
        data.at(time).at(0) = 0;
        for (int i=0; i<domsize; ++i) {
            dtype d1 = std::max(backward.at(0)->data.at(time).at(i), 1e-10);
            data.at(time).at(0) -= backward.at(1)->data.at(time).at(i) * std::log(d1);
        }
    }

    void CEE::calcPartialDerivative()
    {
        for (int i=0; i<domsize; ++i) {
            dtype d1 = std::max(backward.at(0)->data.at(time).at(i), 1e-10);
            backward.at(0)->grad.at(time).at(i) -= backward.at(1)->data.at(time).at(i) / d1 * grad.at(time).at(0);
            backward.at(1)->grad.at(time).at(i) -= std::log(backward.at(0)->data.at(time).at(i)) * grad.at(time).at(0);
        }
    }

    
    
    ReLU::ReLU (Node *node1) : MtoM (node1){};

    void ReLU::calcData()
    {
        for (int i=0; i<domsize; ++i) {
            data.at(time).at(i) = (backward.at(0)->data.at(time).at(i) >= 0) ? backward.at(0)->data.at(time).at(i) : 0;
        }
    }

    void ReLU::calcPartialDerivative()
    {
        for (int i=0; i<domsize; ++i) {
            backward.at(0)->grad.at(time).at(i) += (backward.at(0)->data.at(time).at(i) >= 0) ? 1 * grad.at(time).at(i) : 0;
        }
    }



    Sigmoid::Sigmoid (Node *node1) : MtoM (node1){};

    void Sigmoid::calcData()
    {
        for (int i=0; i<domsize; ++i) {
            dtype x = std::min(10.0, std::max(-10.0, backward.at(0)->data.at(time).at(i)));
            data.at(time).at(i) = 1 / (1 + std::exp(-x));
        }
    }

    void Sigmoid::calcPartialDerivative()
    {
        for (int i=0; i<domsize; ++i) {
            backward.at(0)->grad.at(time).at(i) = data.at(time).at(i) * (1 - (data.at(time).at(i))) * grad.at(time).at(i);
        }
    }


    Tanh::Tanh (Node *node1) : MtoM (node1){};

    void Tanh::calcData()
    {
        for (int i=0; i<domsize; ++i) {
            dtype x = std::min(10.0, std::max(-10.0, backward.at(0)->data.at(time).at(i)));
            dtype e2x = std::exp(2 * x);
            data.at(time).at(i) = (e2x - 1) / (e2x + 1);
        }
    }

    void Tanh::calcPartialDerivative()
    {
        for (int i=0; i<domsize; ++i) {
            backward.at(0)->grad.at(time).at(i) = (1 - (data.at(time).at(i)) * (data.at(time).at(i))) * grad.at(time).at(i);
        }
    }



    Softmax::Softmax (Node *node1) : MtoM (node1){}

    void Softmax::calcData()
    {
        dtype max = backward.at(0)->data.at(time).at(0);
        for (int i=1; i<domsize; ++i) {
            max = std::max(max, backward.at(0)->data.at(time).at(i));
        }

        dtype sum = 0;
        for (int i=0; i<domsize; ++i) {
            dtype z = std::max(backward.at(0)->data.at(time).at(i)-max, -10.0);
            sum += std::exp(z);
        }

        for (int i=0; i<domsize; ++i) {
            dtype z = std::max(backward.at(0)->data.at(time).at(i)-max, -10.0);
            data.at(time).at(i) = std::exp(z) / sum;
        }
    }

    void Softmax::calcPartialDerivative() 
    {
        for (int i=0; i<domsize; ++i) {
            for (int j=0; j<domsize; ++j) {
                if (i == j) {
                    backward.at(0)->grad.at(time).at(i) += data.at(time).at(j) * (1 - data.at(time).at(i)) * grad.at(time).at(j);
                } else {
                    backward.at(0)->grad.at(time).at(i) -= data.at(time).at(j) *      data.at(time).at(i)  * grad.at(time).at(j);
                }
            }
        }
    }



    Norm2::Norm2 (Node *node1) : Mto1 (node1){};

    void Norm2::calcData()
    {
        data.at(time).at(0) = 0;
        for (int i=0; i<domsize; ++i) {
            data.at(time).at(0) += backward.at(0)->data.at(time).at(i) * backward.at(0)->data.at(time).at(i);
        }
        data.at(time).at(0) = sqrt(data.at(time).at(0));
    }
            
    void Norm2::calcPartialDerivative()
    {
        for (int i=0; i<domsize; ++i) {
            backward.at(0)->grad.at(time).at(i) += backward.at(0)->data.at(time).at(i) / data.at(time).at(0) * grad.at(time).at(0);
        }
    }

    
    
    Affine::Affine (Node *node1, vec2<dtype> Weight, dtype bias)
    : Node (node1->dsize, Weight.at(0).size(), 1), bias(bias)
    {
        assert (node1->dsize + 1 == Weight.size());
        for (int i=1; i<Weight.size(); ++i) {
            assert (Weight.at(i).size() == Weight.at(0).size());
        }
        assert (node1->width == 1);
        
        weight = Weight;

        gradWeight.resize(domsize+1);
        for (int i=0; i<=domsize; ++i) {
            gradWeight.at(i).resize(height);
        }

        backward.resize(1);
        backward.at(0) = node1;

        pushThis(node1);
    }

    Affine::Affine (Node *node1, vec2<dtype> Weight)
    : Affine (node1, Weight, 1){}

    void Affine::calcData()
    {
        for (int i=0; i<dsize; ++i) {
            data.at(time).at(i) = 0;
            for (int j=0; j<domsize; ++j) {
                data.at(time).at(i) += weight.at(j).at(i) * backward.at(0)->data.at(time).at(j);
            }
            data.at(time).at(i) += weight.at(domsize).at(i) * bias;
        }
    }

    void Affine::calcPartialDerivative()
    {
        for (int i=0; i<domsize; ++i) {
            for (int j=0; j<dsize; ++j) {
                backward.at(0)->grad.at(time).at(i) += weight.at(i).at(j) * grad.at(time).at(j);
            }
        }

        for (int i=0; i<domsize; ++i) {
            for (int j=0; j<dsize; ++j) {
                gradWeight.at(i).at(j) += backward.at(0)->data.at(time).at(i) * grad.at(time).at(j);
            }
        }
        for (int j=0; j<dsize; ++j) {
            gradWeight.at(domsize).at(j) += bias * grad.at(time).at(j);
        }
    }

    void Affine::updateParameters(dtype eta)
    {
        for (int i=0; i<=domsize; ++i) {
            for (int j=0; j<dsize; ++j) {
                weight.at(i).at(j) -= eta * gradWeight.at(i).at(j);
                gradWeight.at(i).at(j) = 0;
            }
        }
    }



    Convolution2d::Convolution2d (vec1<Node*> nodes, vec3<dtype> Kernel, dtype bias, size_t stride, size_t topPadding, size_t leftPadding, size_t height, size_t width)
    : Filter2d (nodes, Kernel.at(0).size(), Kernel.at(0).at(0).size(), stride, topPadding, leftPadding, height, width)
    {   
        assert (Kernel.size() == nodes.size());
        for(int c=0; c<Kernel.size(); ++c) {
            assert (Kernel.at(c).size() == kheight);
            for (int i=0; i<kheight; ++i) {
                assert (Kernel.at(c).at(i).size() == kwidth);
            }
        }

        kernel = Kernel;

        this->bias = bias;
        gradBias = 0;

        gradKernel.resize(backward.size());
        for (int c=0; c<backward.size(); ++c) {
            gradKernel.at(c).resize(kheight);
            for (int i=0; i<kheight; ++i) {
                gradKernel.at(c).at(i).resize(kwidth);
            }
        }
    }

    Convolution2d::Convolution2d (vec1<Node*> nodes, vec3<dtype> Kernel, dtype bias, size_t stride, size_t height, size_t width)
    : Convolution2d (nodes, Kernel, bias, stride, (stride*(height-1) + Kernel.at(0).size() - nodes.at(0)->height)/2, (stride*(width-1) + Kernel.at(0).at(0).size() - nodes.at(0)->width)/2, height, width)
    {
        assert (stride * (height - 1) + Kernel.at(0).size()       >= nodes.at(0)->height);
        assert (stride * (width  - 1) + Kernel.at(0).at(0).size() >= nodes.at(0)->width);
    }

    Convolution2d::Convolution2d (vec1<Node*> nodes, vec3<dtype> Kernel, dtype bias, size_t stride)
    : Convolution2d (nodes, Kernel, bias, stride, (nodes.at(0)->height - Kernel.at(0).size() + (stride-1))/stride + 1, (nodes.at(0)->width - Kernel.at(0).at(0).size() + (stride-1))/stride + 1){}

    Convolution2d::Convolution2d (vec1<Node*> nodes, vec3<dtype> Kernel, dtype bias, size_t height, size_t width)
    : Convolution2d (nodes, Kernel, bias, 1, height, width){}

    Convolution2d::Convolution2d (vec1<Node*> nodes, vec3<dtype> Kernel, dtype bias)
    : Convolution2d (nodes, Kernel, bias, 1, nodes.at(0)->height - (Kernel.at(0).size()-1), nodes.at(0)->width - (Kernel.at(0).at(0).size()-1))
    {
        assert (Kernel.at(0).size()       <= nodes.at(0)->height);
        assert (Kernel.at(0).at(0).size() <= nodes.at(0)->width);
    }

    void Convolution2d::calcData()
    {    
        size_t bheight = backward.at(0)->height;
        size_t bwidth  = backward.at(0)->width;
        for (int a=0; a<height; ++a) {
            for (int b=0; b<width; ++b) {
                int index = a * width + b;
                data.at(time).at(index) = bias;
                for (int c=0; c<backward.size(); ++c) {
                    for (int i=0; i<kheight; ++i) {
                        for (int j=0; j<kwidth; ++j) {
                            data.at(time).at(index) += kernel.at(c).at(i).at(j) * getDomData(c, a * sw + i - pt, b * sw + j - pl);
                        }
                    }
                }
            }
        }
    }

    void Convolution2d::calcPartialDerivative()
    {
        size_t bheight = backward.at(0)->height;
        size_t bwidth  = backward.at(0)->width;
        for (int c=0; c<backward.size(); ++c) {
            for (int a=0; a<bheight; ++a) {
                for (int b=0; b<bwidth; ++b) {
                    //backward.at(0)->grad.at(0).at(a * bwidth + b) = 0;
                    for (int i=(a+pt)%sw; i<kheight; i+=sw) {
                        for (int j=(b+pl)%sw; j<kwidth; j+=sw) {
                            int col = (a - i + pt) / sw;
                            int row = (b - j + pl) / sw;
                            if (0 <= col && col < height && 0 <= row && row < width) {
                                backward.at(c)->grad.at(time).at(a * bwidth + b) += kernel.at(c).at(i).at(j) * grad.at(time).at(col * width + row);
                            }
                        }
                    }
                }
            }
        }
        for (int c=0; c<backward.size(); ++c) {
            for (int i=0; i<kheight; ++i) {
                for (int j=0; j<kwidth; ++j) {
                    for (int a=0; a<height; ++a) {
                        for (int b=0; b<width; ++b) {
                            gradKernel.at(c).at(i).at(j) += getDomData(c, a * sw + i - pt, b * sw + j - pl) * grad.at(time).at(a * width + b);
                        }
                    }
                    //std::cout << "grad : " << gradKernel.at(c).at(i).at(j) << std::endl;
                }
            }
        }
        for (int a=0; a<height; ++a) {
            for (int b=0; b<width; ++b) {
                gradBias += grad.at(time).at(a * width + b);
            }
        }
        //std::cout << "bias : " << gradBias << std::endl;
    }

    void Convolution2d::updateParameters(dtype eta)
    {
        for (int c=0; c<backward.size(); ++c) {
            for (int i=0; i<kheight; ++i) {
                for (int j=0; j<kwidth; ++j) {
                    kernel.at(c).at(i).at(j) -= eta * gradKernel.at(c).at(i).at(j);
                    gradKernel.at(c).at(i).at(j) = 0;
                }
            }
        }
        bias -= eta * gradBias;
        gradBias = 0;
    }



    MaxPooling2d::MaxPooling2d (Node *node1, size_t kernelHeight, size_t kernelWidth, size_t stride, size_t topPadding, size_t leftPadding, size_t height, size_t width)
    : Filter2d ({node1}, kernelHeight, kernelWidth, stride, topPadding, leftPadding, height, width)
    {   
        int bottomPadding = stride * (height - 1) + kernelHeight - node1->height - topPadding;
        int rightPadding  = stride * (width  - 1) + kernelWidth  - node1->width  - leftPadding;
        assert (topPadding  < kernelHeight && bottomPadding < kernelHeight);
        assert (leftPadding < kernelWidth  && rightPadding  < kernelWidth);

        maxCount.resize(this->height * this->width);
    }

    MaxPooling2d::MaxPooling2d (Node *node1, size_t kernelHeight, size_t kernelWidth, size_t stride, size_t height, size_t width)
    : MaxPooling2d (node1, kernelHeight, kernelWidth, stride, (stride*(height-1) + kernelHeight - node1->height)/2, (stride*(width-1) + kernelWidth - node1->width)/2, height, width){}

    MaxPooling2d::MaxPooling2d (Node *node1, size_t kernelHeight, size_t kernelWidth, size_t stride)
    : MaxPooling2d (node1, kernelHeight, kernelWidth, stride, (node1->height - kernelHeight + (stride-1))/stride + 1, (node1->width - kernelWidth + (stride-1))/stride + 1){}

    void MaxPooling2d::calcData()
    {
        size_t bheight = backward.at(0)->height;
        size_t bwidth  = backward.at(0)->width;
        for (int a=0; a<height; ++a) {
            for (int b=0; b<width; ++b) {
                int count = 0;
                dtype max = std::nan("");
                for (int i=0; i<kheight; ++i) {
                    for (int j=0; j<kwidth; ++j) {
                        int col = a * sw + i - pt;
                        int row = b * sw + j - pl;
                        if (!inDomain(col, row)) {
                            continue;
                        }
                        dtype d = getDomData(col, row);
                        if (std::isnan(max) || max < d) {
                            max = d;
                            count = 1;
                        } else if (max == d) {
                            ++count;
                        }
                    }
                }
                assert (!std::isnan(max));
                data.at(time).at(a * width + b) = max;
                maxCount.at(a * width + b) = count;
            }
        }
    }

    void MaxPooling2d::calcPartialDerivative()
    {
        size_t bheight = backward.at(0)->height;
        size_t bwidth  = backward.at(0)->width;
        for (int a=0; a<bheight; ++a) {
            for (int b=0; b<bwidth; ++b) {
                //backward.at(0)->grad.at(0).at(a * bwidth + b) = 0;
                for (int i=(a+pt)%sw; i<kheight; i+=sw) {
                    for (int j=(b+pl)%sw; j<kwidth; j+=sw) {
                        int col = (a - i + pt) / sw;
                        int row = (b - j + pl) / sw;
                        if (   0 <= col && col < height && 0 <= row && row < width
                            && (backward.at(0)->data.at(time).at(a * bwidth + b) == data.at(time).at(col * width + row))) {
                            backward.at(0)->grad.at(time).at(a * bwidth + b) += grad.at(time).at(col * width + row) / maxCount.at(col * width + row);
                        }
                    }
                }
            }
        }
    }



    AveragePooling2d::AveragePooling2d (Node *node1, size_t kernelHeight, size_t kernelWidth, size_t stride, size_t topPadding, size_t leftPadding, size_t height, size_t width)
    : Filter2d ({node1}, kernelHeight, kernelWidth, stride, topPadding, leftPadding, height, width)
    {   
        int bottomPadding = stride * (height - 1) + kernelHeight - node1->height - topPadding;
        int rightPadding  = stride * (width  - 1) + kernelWidth  - node1->width  - leftPadding;
        assert (topPadding  < kernelHeight && bottomPadding < kernelHeight);
        assert (leftPadding < kernelWidth  && rightPadding  < kernelWidth);
    }

    AveragePooling2d::AveragePooling2d (Node *node1, size_t kernelHeight, size_t kernelWidth, size_t stride, size_t height, size_t width)
    : AveragePooling2d (node1, kernelHeight, kernelWidth, stride, (stride*(height-1) + kernelHeight - node1->height)/2, (stride*(width-1) + kernelWidth - node1->width)/2, height, width){}

    AveragePooling2d::AveragePooling2d (Node *node1, size_t kernelHeight, size_t kernelWidth, size_t stride)
    : AveragePooling2d (node1, kernelHeight, kernelWidth, stride, (node1->height - kernelHeight + (stride-1))/stride + 1, (node1->width - kernelWidth + (stride-1))/stride + 1){}

    void AveragePooling2d::calcData()
    {
        size_t bheight = backward.at(0)->height;
        size_t bwidth  = backward.at(0)->width;
        for (int a=0; a<height; ++a) {
            for (int b=0; b<width; ++b) {
                dtype sum = 0;
                for (int i=0; i<kheight; ++i) {
                    for (int j=0; j<kwidth; ++j) {
                        int col = a * sw + i - pt;
                        int row = b * sw + j - pl;
                        sum += getDomData(col, row);
                    }
                }
                data.at(time).at(a * width + b) = sum / (kheight * kwidth);
            }
        }
    }

    void AveragePooling2d::calcPartialDerivative()
    {
        size_t bheight = backward.at(0)->height;
        size_t bwidth  = backward.at(0)->width;
        for (int a=0; a<bheight; ++a) {
            for (int b=0; b<bwidth; ++b) {
                //backward.at(0)->grad.at(0).at(a * bwidth + b) = 0;
                for (int i=(a+pt)%sw; i<kheight; i+=sw) {
                    for (int j=(b+pl)%sw; j<kwidth; j+=sw) {
                        int col = (a - i + pt) / sw;
                        int row = (b - j + pl) / sw;
                        if (   0 <= col && col < height && 0 <= row && row < width) {
                            backward.at(0)->grad.at(time).at(a * bwidth + b) += grad.at(time).at(col * width + row) / (kheight * kwidth);
                        }
                    }
                }
            }
        }
    }



    size_t getSumSizeOfData(vec1<Node*> nodes)
    {
        size_t ret = 0;
        for (int i=0; i<nodes.size(); ++i) {
            ret += nodes.at(i)->dsize;
        }
        return ret;
    }

    size_t getSumSizeOfHeight(vec1<Node*> nodes)
    {
        size_t ret = 0;
        for (int i=0; i<nodes.size(); ++i) {
            ret += nodes.at(i)->height;
        }
        return ret;
    }

    void dumpNode(Node const node1, std::string name, ttype time)
    {
        std::cout << name << " back size = " << node1.backward.size() << std::endl;
        std::cout << name << " forw size = " << node1.forward.size() << std::endl;
        std::cout << name << " data size = " << node1.dsize << std::endl;
        std::cout << name << " data      = ";
        for (int i=0; i<node1.dsize; ++i) { std::cout << node1.data.at(time).at(i) << ((i==node1.dsize-1) ? "" : " "); }
        std::cout << std::endl;
        std::cout << name << " grad size = " << node1.dsize << std::endl;
        std::cout << name << " grad      = ";
        for (int i=0; i<node1.dsize; ++i) { std::cout << node1.grad.at(time).at(i) << ((i==node1.dsize-1) ? "" : " "); }
        std::cout << std::endl;
    }
    void dumpNode(Node const node1, std::string name)
    {
        dumpNode(node1, name, 0);
    }
};
