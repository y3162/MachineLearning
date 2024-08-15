#include <iostream>
#include <cassert>
#include <vector>

namespace CG
{
    template<typename T> using vec1 = std::vector<T>;
    template<typename T> using vec2 = vec1<vec1<T>>;
    using dtype = double;

    class Node
    {
        public :
            vec1<dtype> data;
            vec1<dtype> grad;
            vec1<Node*> forward;
            vec1<Node*> backward;
            size_t      bsize;
            int         f_count = 0;
            int         b_count = 0;

            Node (){}

            void pushThis(Node *node) // push this as argument's forward node
            {
                size_t fsize = node->forward.size();
                node->forward.resize(fsize+1);
                node->forward.at(fsize) = this;
            }

            virtual void calcData(){}
            void forwardPropagation()
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

            virtual void calcPartialDerivative(){}
            void backwardPropagation()
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

            virtual void updateParameters(dtype eta){}
            void update(dtype eta)
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
    };

    class Leaf : public Node
    {
        public :
            Leaf (size_t size)
            {
                bsize = 0;
                data.resize(size);
                grad.resize(size);
                forward.resize(0);
                backward.resize(0);
            }

            void getInput(vec1<dtype> input)
            {
                assert (data.size() == input.size());
                data = input;
            }
    };

    class Add : public Node
    {   
        public :
            Add (Node *node1, Node *node2)
            {   
                assert (node1->data.size() == node2->data.size());

                bsize = node1->data.size();
                data.resize(bsize);
                grad.resize(bsize);

                forward.resize(0);

                backward.resize(2);
                backward.at(0) = node1;
                backward.at(1) = node2;

                pushThis(node1);
                pushThis(node2);
            };

            virtual void calcData()
            {
                for (int i=0; i<bsize; ++i) {
                    data.at(i) = backward.at(0)->data.at(i) + backward.at(1)->data.at(i);
                }
            }

            virtual void calcPartialDerivative()
            {   
                for (int i=0; i<bsize; ++i) {
                    backward.at(0)->grad.at(i) += 1 * grad.at(i);
                    backward.at(1)->grad.at(i) += 1 * grad.at(i);
                }
            }
    };

    class Sub : public Add
    {
        public :
            Sub (Node *node1, Node *node2) : Add (node1, node2){};

            virtual void calcData()
            {
                for (int i=0; i<bsize; ++i) {
                    data.at(i) = backward.at(0)->data.at(i) - backward.at(1)->data.at(i);
                }
            }

            virtual void calcPartialDerivative()
            {
                for (int i=0; i<bsize; ++i) {
                    backward.at(0)->grad.at(i) +=  1 * grad.at(i);
                    backward.at(1)->grad.at(i) += -1 * grad.at(i);
                }
            }
    };

    class Dots : public Node
    {
        public :
            Dots (Node *node1, Node *node2)
            {   
                assert (node1->data.size() == node2->data.size());

                bsize = node1->data.size();
                data.resize(1);
                grad.resize(1);

                forward.resize(0);

                backward.resize(2);
                backward.at(0) = node1;
                backward.at(1) = node2;

                pushThis(node1);
                pushThis(node2);
            };

            virtual void calcData()
            {   
                data.at(0) = 0;
                for (int i=0; i<bsize; ++i) {
                    data.at(0) += backward.at(0)->data.at(i) * backward.at(1)->data.at(i);
                }
            }

            virtual void calcPartialDerivative()
            {
                for (int i=0; i<bsize; ++i) {
                    backward.at(0)->grad.at(i) += backward.at(1)->data.at(i) * grad.at(0);
                    backward.at(1)->grad.at(i) += backward.at(0)->data.at(i) * grad.at(0);
                }
            }
    };

    class Affine : public Node
    {
        public :
            vec2<dtype> Weight;
            vec2<dtype> gradWeight;
            dtype       bias = 1;

            Affine (Node *node1, vec2<dtype> W)
            {
                assert (node1->data.size() + 1 == W.size());
                
                Weight = W;

                bsize = node1->data.size();
                size_t osize = W.at(0).size();
                gradWeight.resize(bsize+1);
                for (int i=0; i<=bsize; ++i) {
                    gradWeight.at(i).resize(osize);
                }
                data.resize(osize);
                grad.resize(osize);

                forward.resize(0);

                backward.resize(1);
                backward.at(0) = node1;

                pushThis(node1);
            }

            virtual void calcData()
            {
                for (int i=0; i<data.size(); ++i) {
                    data.at(i) = 0;
                    for (int j=0; j<bsize; ++j) {
                        data.at(i) += Weight.at(j).at(i) * backward.at(0)->data.at(j);
                    }
                    data.at(i) += Weight.at(bsize).at(i) * bias;
                }
            }

            virtual void calcPartialDerivative()
            {
                for (int i=0; i<bsize; ++i) {
                    for (int j=0; j<data.size(); ++j) {
                        backward.at(0)->grad.at(i) += Weight.at(i).at(j) * grad.at(j);
                    }
                }

                for (int i=0; i<bsize; ++i) {
                    for (int j=0; j<data.size(); ++j) {
                        gradWeight.at(i).at(j) += backward.at(0)->data.at(i) * grad.at(j);
                    }
                }
                for (int j=0; j<data.size(); ++j) {
                    gradWeight.at(bsize).at(j) += grad.at(j);
                }
            }

            virtual void updateParameters(dtype eta)
            {
                for (int i=0; i<=bsize; ++i) {
                    for (int j=0; j<data.size(); ++j) {
                        Weight.at(i).at(j) -= eta * gradWeight.at(i).at(j);
                        gradWeight.at(i).at(j) = 0;
                    }
                }
            }
    };

    class ReLU : public Node{
        public :
            ReLU (Node *node1)
            {   
                bsize = node1->data.size();
                data.resize(bsize);
                grad.resize(bsize);

                forward.resize(0);

                backward.resize(1);
                backward.at(0) = node1;

                pushThis(node1);
            };

            virtual void calcData()
            {
                for (int i=0; i<bsize; ++i) {
                    data.at(i) = (backward.at(0)->data.at(i) >= 0) ? backward.at(0)->data.at(i) : 0;
                }
            }

            virtual void calcPartialDerivative()
            {
                for (int i=0; i<bsize; ++i) {
                    backward.at(0)->grad.at(i) += (backward.at(0)->data.at(i) >= 0) ? 1 * grad.at(i) : 0;
                }
            }
    };

    class Softmax : public ReLU
    {
        public : 
            Softmax (Node *node1) : ReLU (node1){}

            virtual void calcData()
            {
                dtype max = backward.at(0)->data.at(0);
                for (int i=1; i<bsize; ++i) {
                    max = std::max(max, backward.at(0)->data.at(i));
                }

                dtype sum = 0;
                for (int i=0; i<bsize; ++i) {
                    dtype z = std::max(backward.at(0)->data.at(i)-max, -200.0);
                    sum += std::exp(z);
                }

                for (int i=0; i<bsize; ++i) {
                    dtype z = std::max(backward.at(0)->data.at(i)-max, -200.0);
                    data.at(i) = std::exp(z) / sum;
                }
            }

            virtual void calcPartialDerivative() 
            {
                for (int i=0; i<bsize; ++i) {
                    for (int j=0; j<bsize; ++j) {
                        if (i == j) 
                            backward.at(0)->grad.at(i) += (1 - data.at(i)) * data.at(j) * grad.at(j);
                        else
                            backward.at(0)->grad.at(i) +=    - data.at(i)  * data.at(j) * grad.at(j);
                    }
                }
            }
    };

    class Norm2 : public Node
    {
        public :
            Norm2 (Node *node1)
            {   
                bsize = node1->data.size();
                data.resize(1);
                grad.resize(1);

                forward.resize(0);

                backward.resize(1);
                backward.at(0) = node1;

                pushThis(node1);
            };

            virtual void calcData()
            {
                data.at(0) = 0;
                for (int i=0; i<bsize; ++i) {
                    data.at(0) += backward.at(0)->data.at(i) * backward.at(0)->data.at(i);
                }
                data.at(0) = sqrt(data.at(0));
            }
            
            virtual void calcPartialDerivative()
            {
                for (int i=0; i<bsize; ++i) {
                    backward.at(0)->grad.at(i) += backward.at(0)->data.at(i) / data.at(0) * grad.at(0);
                }
            }
    };

    class MLE : public Dots
    {
        public : 
            MLE (Node *node1, Node *node2) : Dots (node1, node2){};

            virtual void calcData()
            {   
                data.at(0) = 0;
                for (int i=0; i<bsize; ++i) {
                    dtype err = backward.at(0)->data.at(i) - backward.at(1)->data.at(i);
                    data.at(0) += err * err;
                }
                data.at(0) /= bsize;
            }

            virtual void calcPartialDerivative()
            {
                for (int i=0; i<bsize; ++i) {
                    dtype err = backward.at(0)->data.at(i) - backward.at(1)->data.at(i);
                    backward.at(0)->grad.at(i) +=   2 * err * grad.at(0) / bsize;
                    backward.at(1)->grad.at(i) += - 2 * err * grad.at(0) / bsize;
                }
            }      
    };

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
