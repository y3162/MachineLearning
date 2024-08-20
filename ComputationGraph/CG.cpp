#include <iostream>
#include <cassert>
#include <vector>
#include "CG.hpp"
#include "Type.hpp"

namespace CG
{
    Node::Node (){}

    void Node::pushThis(Node *node) // push this as argument's forward node
    {
        size_t fsize = node->forward.size();
        node->forward.resize(fsize+1);
        node->forward.at(fsize) = this;
    }

    void Node::calcData(){}
    void Node::forwardPropagation()
    {
        if (++b_count < backward.size()) {
            return;
        } else { // When all the forward passes from the units in the preceding layer have been completed
            b_count = 0;
        }

        for (int i=0; i<grad.size(); ++i) {
            grad.at(i) = 0;
        }

        calcData();

        for (int i=0; i<forward.size(); ++i) {
            forward.at(i)->forwardPropagation();
        }
    }

    void Node::calcPartialDerivative(){}
    void Node::backwardPropagation()
    {   
        if (++f_count < forward.size()) {
            return;
        } else { // When all the backpropagations from the units in the next layer have been completed
            f_count = 0;
        }

        if (forward.size() == 0) {
            assert (data.size() == 1);
            grad.at(0) = 1;
        }
        calcPartialDerivative();

        for (int i=0; i<backward.size(); ++i) {
            backward.at(i)->backwardPropagation();
        }
    }

    void Node::updateParameters(dtype eta){}
    void Node::update(dtype eta)
    {
        if (++f_count < forward.size()) {
            return;
        } else { // When all the backpropagations from the units in the next layer have been completed
            f_count = 0;
        }

        updateParameters(eta);

        for (int i=0; i<backward.size(); ++i) {
            backward.at(i)->update(eta);
        }
    }



    Leaf1::Leaf1 (size_t size)
    {
        domsize = 0;
        height  = size;
        width   = 1;  
        data.resize(size);
        grad.resize(size);
        forward.resize(0);
        backward.resize(0);
    }

    void Leaf1::getInput(const vec1<dtype> input)
    {
        assert (data.size() == input.size());
        data = input;
    }



    Leaf2::Leaf2 (size_t height, size_t width)
    {
        domsize = 0;
        this->height = height;
        this->width  = width;
        data.resize(height * width);
        grad.resize(height * width);
        forward.resize(0);
        backward.resize(0);
    }

    void Leaf2::getInput(const vec1<dtype> input)
    {
        assert (data.size() == input.size());
        data = input;
    }

    void Leaf2::getInput(const vec2<dtype> input)
    {
        assert (input.size() == height);
        assert (input.at(0).size() == width);

        for (int i=0; i<height; ++i) {
            for (int j=0; j<width; ++j) {
                data.at(i * width + j) = input.at(i).at(j);
            }
        }
    }



    MMtoM::MMtoM (Node *node1, Node *node2)
    {   
        assert (node1->data.size() == node2->data.size());
        assert (node1->height == node2->height);
        assert (node1->width  == node2->width);

        domsize = node1->data.size();
        height = node1->height;
        width  = node2->width;
        data.resize(domsize);
        grad.resize(domsize);

        forward.resize(0);

        backward.resize(2);
        backward.at(0) = node1;
        backward.at(1) = node2;

        pushThis(node1);
        pushThis(node2);
    };

    
    
    MMto1::MMto1 (Node *node1, Node *node2)
    {   
        assert (node1->data.size() == node2->data.size());
        assert (node1->height == node2->height);
        assert (node1->width  == node2->width);

        domsize = node1->data.size();
        height = 1;
        width  = 1;
        data.resize(1);
        grad.resize(1);

        forward.resize(0);

        backward.resize(2);
        backward.at(0) = node1;
        backward.at(1) = node2;

        pushThis(node1);
        pushThis(node2);
    };



    MtoM::MtoM (Node *node1)
    {   
        domsize = node1->data.size();
        height  = node1->height;
        width   = node1->width;
        data.resize(domsize);
        grad.resize(domsize);

        forward.resize(0);

        backward.resize(1);
        backward.at(0) = node1;

        pushThis(node1);
    };



    Mto1::Mto1 (Node *node1)
    {   
        domsize = node1->data.size();
        height  = 1;
        width   = 1;
        data.resize(1);
        grad.resize(1);

        forward.resize(0);

        backward.resize(1);
        backward.at(0) = node1;

        pushThis(node1);
    };



    Add::Add (Node *node1, Node *node2) : MMtoM (node1, node2){};

    void Add::calcData()
    {
        for (int i=0; i<domsize; ++i) {
            data.at(i) = backward.at(0)->data.at(i) + backward.at(1)->data.at(i);
        }
    }

    void Add::calcPartialDerivative()
    {   
        for (int i=0; i<domsize; ++i) {
            backward.at(0)->grad.at(i) += 1 * grad.at(i);
            backward.at(1)->grad.at(i) += 1 * grad.at(i);
        }
    }



    Sub::Sub (Node *node1, Node *node2) : MMtoM (node1, node2){};

    void Sub::calcData()
    {
        for (int i=0; i<domsize; ++i) {
            data.at(i) = backward.at(0)->data.at(i) - backward.at(1)->data.at(i);
        }
    }

    void Sub::calcPartialDerivative()
    {
        for (int i=0; i<domsize; ++i) {
            backward.at(0)->grad.at(i) +=  1 * grad.at(i);
            backward.at(1)->grad.at(i) += -1 * grad.at(i);
        }
    }



    Dots::Dots (Node *node1, Node *node2) : MMto1 (node1, node2){};

    void Dots::calcData()
    {   
        data.at(0) = 0;
        for (int i=0; i<domsize; ++i) {
            data.at(0) += backward.at(0)->data.at(i) * backward.at(1)->data.at(i);
        }
    }

    void Dots::calcPartialDerivative()
    {
        for (int i=0; i<domsize; ++i) {
            backward.at(0)->grad.at(i) += backward.at(1)->data.at(i) * grad.at(0);
            backward.at(1)->grad.at(i) += backward.at(0)->data.at(i) * grad.at(0);
        }
    }



    MSE::MSE (Node *node1, Node *node2) : MMto1 (node1, node2){};

    void MSE::calcData()
    {   
        data.at(0) = 0;
        for (int i=0; i<domsize; ++i) {
            dtype err = backward.at(0)->data.at(i) - backward.at(1)->data.at(i);
            data.at(0) += err * err;
        }
        data.at(0) /= domsize;
    }

    void MSE::calcPartialDerivative()
    {
        for (int i=0; i<domsize; ++i) {
            dtype err = backward.at(0)->data.at(i) - backward.at(1)->data.at(i);
            backward.at(0)->grad.at(i) +=   2 * err * grad.at(0) / domsize;
            backward.at(1)->grad.at(i) += - 2 * err * grad.at(0) / domsize;
        }
    }



    CEE::CEE (Node *node1, Node *node2) : MMto1 (node1, node2){};

    void CEE::calcData()
    {
        data.at(0) = 0;
        for (int i=0; i<domsize; ++i) {
            dtype d1 = std::max(backward.at(0)->data.at(i), 1e-200);
            data.at(0) -= backward.at(1)->data.at(i) * std::log(d1);
        }
    }

    void CEE::calcPartialDerivative()
    {
        for (int i=0; i<domsize; ++i) {
            dtype d1 = std::max(backward.at(0)->data.at(i), 1e-200);
            backward.at(0)->grad.at(i) -= backward.at(1)->data.at(i) / d1 * grad.at(0);
            backward.at(1)->grad.at(i) -= std::log(backward.at(0)->data.at(i)) * grad.at(0);
        }
    }

    
    
    ReLU::ReLU (Node *node1) : MtoM (node1){};

    void ReLU::calcData()
    {
        for (int i=0; i<domsize; ++i) {
            data.at(i) = (backward.at(0)->data.at(i) >= 0) ? backward.at(0)->data.at(i) : 0;
        }
    }

    void ReLU::calcPartialDerivative()
    {
        for (int i=0; i<domsize; ++i) {
            backward.at(0)->grad.at(i) += (backward.at(0)->data.at(i) >= 0) ? 1 * grad.at(i) : 0;
        }
    }



    Softmax::Softmax (Node *node1) : MtoM (node1){}

    void Softmax::calcData()
    {
        dtype max = backward.at(0)->data.at(0);
        for (int i=1; i<domsize; ++i) {
            max = std::max(max, backward.at(0)->data.at(i));
        }

        dtype sum = 0;
        for (int i=0; i<domsize; ++i) {
            dtype z = std::max(backward.at(0)->data.at(i)-max, -200.0);
            sum += std::exp(z);
        }

        for (int i=0; i<domsize; ++i) {
            dtype z = std::max(backward.at(0)->data.at(i)-max, -200.0);
            data.at(i) = std::exp(z) / sum;
        }
    }

    void Softmax::calcPartialDerivative() 
    {
        for (int i=0; i<domsize; ++i) {
            for (int j=0; j<domsize; ++j) {
                if (i == j) {
                    backward.at(0)->grad.at(i) += (1 - data.at(i)) * data.at(j) * grad.at(j);
                } else {
                    backward.at(0)->grad.at(i) +=    - data.at(i)  * data.at(j) * grad.at(j);
                }
            }
        }
    }



    Norm2::Norm2 (Node *node1) : Mto1 (node1){};

    void Norm2::calcData()
    {
        data.at(0) = 0;
        for (int i=0; i<domsize; ++i) {
            data.at(0) += backward.at(0)->data.at(i) * backward.at(0)->data.at(i);
        }
        data.at(0) = sqrt(data.at(0));
    }
            
    void Norm2::calcPartialDerivative()
    {
        for (int i=0; i<domsize; ++i) {
            backward.at(0)->grad.at(i) += backward.at(0)->data.at(i) / data.at(0) * grad.at(0);
        }
    }

    
    
    Affine::Affine (Node *node1, const vec2<dtype> W)
    {
        assert (node1->data.size() + 1 == W.size());
        assert (node1->width == 1);
        
        weight = W;

        domsize = node1->data.size();
        size_t osize = W.at(0).size();
        height = osize;
        width  = 1;
        gradweight.resize(domsize+1);
        for (int i=0; i<=domsize; ++i) {
            gradweight.at(i).resize(osize);
        }
        data.resize(osize);
        grad.resize(osize);

        forward.resize(0);

        backward.resize(1);
        backward.at(0) = node1;

        pushThis(node1);
    }

    void Affine::calcData()
    {
        for (int i=0; i<data.size(); ++i) {
            data.at(i) = 0;
            for (int j=0; j<domsize; ++j) {
                data.at(i) += weight.at(j).at(i) * backward.at(0)->data.at(j);
            }
            data.at(i) += weight.at(domsize).at(i) * bias;
        }
    }

    void Affine::calcPartialDerivative()
    {
        for (int i=0; i<domsize; ++i) {
            for (int j=0; j<data.size(); ++j) {
                backward.at(0)->grad.at(i) += weight.at(i).at(j) * grad.at(j);
            }
        }

        for (int i=0; i<domsize; ++i) {
            for (int j=0; j<data.size(); ++j) {
                gradweight.at(i).at(j) += backward.at(0)->data.at(i) * grad.at(j);
            }
        }
        for (int j=0; j<data.size(); ++j) {
            gradweight.at(domsize).at(j) += bias * grad.at(j);
        }
    }

    void Affine::updateParameters(dtype eta)
    {
        for (int i=0; i<=domsize; ++i) {
            for (int j=0; j<data.size(); ++j) {
                weight.at(i).at(j) -= eta * gradweight.at(i).at(j);
                gradweight.at(i).at(j) = 0;
            }
        }
    }



    Convolution::Convolution (Node *node1, const vec2<dtype> K, const size_t padding)
    {
        assert (padding < K.size());
        assert (padding < K.at(0).size());
        
        size_t kheight = K.size();
        size_t kwidth  = K.at(0).size();
        kernel = K;
        gradKernel.resize(kheight);
        for (int i=0; i<kheight; ++i) {
            gradKernel.at(i).resize(kwidth);
        }
        domsize = node1->height * node1->width;
        psize   = padding;
        height  = node1->height + 2 * psize - kheight + 1;
        width   = node1->width  + 2 * psize - kwidth  + 1;

        data.resize(height * width);
        grad.resize(height * width);

        forward.resize(0);

        backward.resize(1);
        backward.at(0) = node1;

        pushThis(node1);
    }

    dtype Convolution::getDomData(int col, int row)
    {
        size_t bheight = backward.at(0)->height;
        size_t bwidth  = backward.at(0)->width;
        if (0 <= col && col < bheight && 0 <= row && row < bwidth) {
            return backward.at(0)->data.at(col * bwidth + row);
        } else {
            return 0;
        }
    }

    void Convolution::calcData()
    {    
        size_t bheight = backward.at(0)->height;
        size_t bwidth  = backward.at(0)->width;
        size_t kheight = kernel.size();
        size_t kwidth  = kernel.at(0).size();
        for (int s=0; s<height; ++s) {
            for (int t=0; t<width; ++t) {
                int index = s * width + t;
                data.at(index) = bias;
                for (int i=0; i<kheight; ++i) {
                    for (int j=0; j<kwidth; ++j) {
                        data.at(index) += kernel.at(i).at(j) * getDomData(s + i - psize, t + j - psize);
                    }
                }
            }
        }
    }

    void Convolution::calcPartialDerivative()
    {
        size_t bheight = backward.at(0)->height;
        size_t bwidth  = backward.at(0)->width;
        size_t kheight = kernel.size();
        size_t kwidth  = kernel.at(0).size();
        for (int s=0; s<bheight; ++s) {
            for (int t=0; t<bwidth; ++t) {
                backward.at(0)->grad.at(s * bwidth + t) = 0;
                for (int i=0; i<kheight; ++i) {
                    for (int j=0; j<kwidth; ++j) {
                        int col = s - i + psize;
                        int row = t - j + psize;
                        if (0 <= col && col < height && 0 <= row && row < width) {
                            backward.at(0)->grad.at(s * bwidth + t) += kernel.at(i).at(j) * grad.at((s - i + psize) * width + (t - j + psize));
                        }
                    }
                }
            }
        }
        for (int i=0; i<kheight; ++i) {
            for (int j=0; j<kwidth; ++j) {
                for (int s=0; s<height; ++s) {
                    for (int t=0; t<width; ++t) {
                        gradKernel.at(i).at(j) += getDomData(s + i - psize, t + j - psize) * grad.at(s * width + t);
                    }
                }
            }
        }
        for (int s=0; s<height; ++s) {
            for (int t=0; t<width; ++t) {
                gradBias += grad.at(s * width + t);
            }
        }
    }

    void Convolution::updateParameters(dtype eta)
    {
        for (int i=0; i<kernel.size(); ++i) {
            for (int j=0; j<kernel.at(0).size(); ++j) {
                kernel.at(i).at(j) -= eta * gradKernel.at(i).at(j);
                gradKernel.at(i).at(j) = 0;
            }
        }
        bias -= eta * gradBias;
        gradBias = 0;
    }

    

    void dumpNode (Node const node1, std::string name)
    {
        std::cout << name << " back size = " << node1.backward.size()    << std::endl;
        std::cout << name << " forw size = " << node1.forward.size() << std::endl;
        std::cout << name << " data size = " << node1.data.size()    << std::endl;
        std::cout << name << " data      = ";
        for (int i=0; i<node1.data.size(); ++i) { std::cout << node1.data.at(i) << ((i==node1.data.size()-1) ? "" : " "); }
        std::cout << std::endl;
        std::cout << name << " grad size = " << node1.grad.size()    << std::endl;
        std::cout << name << " grad      = ";
        for (int i=0; i<node1.grad.size(); ++i) { std::cout << node1.grad.at(i) << ((i==node1.grad.size()-1) ? "" : " "); }
        std::cout << std::endl;
    }
};
