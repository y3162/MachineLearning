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
            vec1<Node*> back;
            size_t      bsize;
            int         f_count = 0;
            int         b_count = 0;

            Node (){}

            virtual void forwardPropagation(){}
            virtual void calcPartialDerivative(dtype eta){}
            void backPropagation(dtype eta)
            {   
                if (eta == 0) return;

                if (++f_count < forward.size()) {
                    return;
                } else { // When all the backpropagations from the units in the next layer have been completed
                    f_count = 0;
                }

                if (forward.size() == 0) {
                    assert (data.size() == 1);
                    grad.at(0) = 1;
                }
                calcPartialDerivative(eta);

                for (int i=0; i<back.size(); ++i) {
                    back.at(i)->backPropagation(eta);
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
                back.resize(0);
            }

            void getInput(vec1<dtype> input)
            {
                assert (data.size() == input.size());
                data = input;
            }

            virtual void forwardPropagation()
            {
                for (int i=0; i<forward.size(); ++i) {
                    forward.at(i)->forwardPropagation();
                }
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

                back.resize(2);
                back.at(0) = node1;
                back.at(1) = node2;

                size_t fsize1 = node1->forward.size();
                node1->forward.resize(fsize1+1);
                node1->forward.at(fsize1) = this;
                size_t fsize2 = node2->forward.size();
                node2->forward.resize(fsize2+1);
                node2->forward.at(fsize2) = this;
            };

            virtual void forwardPropagation()
            {   
                if (++b_count < back.size()) {
                    return;
                } else { // When all the forward passes from the units in the preceding layer have been completed
                    b_count = 0;
                }

                for (int i=0; i<bsize; ++i) {
                    data.at(i) = back.at(0)->data.at(i) + back.at(1)->data.at(i);
                }

                for (int i=0; i<forward.size(); ++i) {
                    forward.at(i)->forwardPropagation();
                }
            }


            virtual void calcPartialDerivative(dtype eta)
            {   
                for (int i=0; i<bsize; ++i) {
                    back.at(0)->grad.at(i) = 1 * grad.at(i);
                    back.at(1)->grad.at(i) = 1 * grad.at(i);
                }
            }
    };

    class Sub : public Add
    {
        public :
            Sub (Node *node1, Node *node2) : Add (node1, node2){};

            virtual void forwardPropagation()
            {   
                if (++b_count < back.size()) {
                    return;
                } else { // When all the forward passes from the units in the preceding layer have been completed
                    b_count = 0;
                }

                for (int i=0; i<bsize; ++i) {
                    data.at(i) = back.at(0)->data.at(i) - back.at(1)->data.at(i);
                }

                for (int i=0; i<forward.size(); ++i) {
                    forward.at(i)->forwardPropagation();
                }
            }

            virtual void calcPartialDerivative(dtype eta)
            {
                for (int i=0; i<bsize; ++i) {
                    back.at(0)->grad.at(i) =  1 * grad.at(i);
                    back.at(1)->grad.at(i) = -1 * grad.at(i);
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

                back.resize(2);
                back.at(0) = node1;
                back.at(1) = node2;

                size_t fsize1 = node1->forward.size();
                node1->forward.resize(fsize1+1);
                node1->forward.at(fsize1) = this;
                size_t fsize2 = node2->forward.size();
                node2->forward.resize(fsize2+1);
                node2->forward.at(fsize2) = this;
            };

            virtual void forwardPropagation()
            {   
                if (++b_count < back.size()) {
                    return;
                } else { // When all the forward passes from the units in the preceding layer have been completed
                    b_count = 0;
                }

                data.at(0) = 0;
                for (int i=0; i<bsize; ++i) {
                    data.at(0) += back.at(0)->data.at(i) * back.at(1)->data.at(i);
                }
                
                for (int i=0; i<forward.size(); ++i) {
                    forward.at(i)->forwardPropagation();
                }
            }

            virtual void calcPartialDerivative(dtype eta)
            {
                for (int i=0; i<bsize; ++i) {
                    back.at(0)->grad.at(i) = back.at(1)->data.at(i) * grad.at(0);
                    back.at(1)->grad.at(i) = back.at(0)->data.at(i) * grad.at(0);
                }
            }
    };

    class Affine : public Node
    {
        public :
            vec2<dtype> Weight;
            dtype       bias = 1;

            Affine (Node *node1, vec2<dtype> W)
            {
                assert (node1->data.size() + 1 == W.size());
                
                Weight = W;

                bsize = node1->data.size();
                size_t osize = W.at(0).size();
                data.resize(osize);
                grad.resize(osize);

                forward.resize(0);

                back.resize(1);
                back.at(0) = node1;

                size_t fsize1 = node1->forward.size();
                node1->forward.resize(fsize1+1);
                node1->forward.at(fsize1) = this;
            }

            virtual void forwardPropagation()
            {
                if (++b_count < back.size()) {
                    return;
                } else { // When all the forward passes from the units in the preceding layer have been completed
                    b_count = 0;
                }

                for (int i=0; i<data.size(); ++i) {
                    data.at(i) = 0;
                    for (int j=0; j<bsize; ++j) {
                        data.at(i) += Weight.at(j).at(i) * back.at(0)->data.at(j);
                    }
                    data.at(i) += Weight.at(bsize).at(i) * bias;
                }
                for (int i=0; i<forward.size(); ++i) {
                    forward.at(i)->forwardPropagation();
                }
            }

            virtual void calcPartialDerivative(dtype eta)
            {
                for (int i=0; i<bsize; ++i) {
                    back.at(0)->grad.at(i) = 0;
                    for (int j=0; j<data.size(); ++j) {
                        back.at(0)->grad.at(i) += Weight.at(i).at(j) * grad.at(j);
                    }
                }

                for (int i=0; i<bsize; ++i) {
                    for (int j=0; j<data.size(); ++j) {
                        Weight.at(i).at(j) -= eta * back.at(0)->data.at(i) * grad.at(j);
                    }
                }
                for (int j=0; j<data.size(); ++j) {
                    Weight.at(bsize).at(j) -= eta * grad.at(j);
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

                back.resize(1);
                back.at(0) = node1;

                size_t fsize1 = node1->forward.size();
                node1->forward.resize(fsize1+1);
                node1->forward.at(fsize1) = this;
            };

            virtual void forwardPropagation()
            {
                if (++b_count < back.size()) {
                    return;
                } else { // When all the forward passes from the units in the preceding layer have been completed
                    b_count = 0;
                }

                data.at(0) = 0;
                for (int i=0; i<bsize; ++i) {
                    data.at(0) += back.at(0)->data.at(i) * back.at(0)->data.at(i);
                }
                data.at(0) = sqrt(data.at(0));

                for (int i=0; i<forward.size(); ++i) {
                    forward.at(i)->forwardPropagation();
                }
            }
            
            virtual void calcPartialDerivative(dtype eta)
            {
                for (int i=0; i<bsize; ++i) {
                    back.at(0)->grad.at(i) = back.at(0)->data.at(i) / data.at(0) * grad.at(0);
                }
            }
    };

    class MLE : public Dots
    {
        public : 
            MLE (Node *node1, Node *node2) : Dots (node1, node2){};

            virtual void forwardPropagation()
            {   
                if (++b_count < back.size()) {
                    return;
                } else { // When all the forward passes from the units in the preceding layer have been completed
                    b_count = 0;
                }

                data.at(0) = 0;
                for (int i=0; i<bsize; ++i) {
                    dtype err = back.at(0)->data.at(i) - back.at(1)->data.at(i);
                    data.at(0) += err * err;
                }
                data.at(0) /= bsize;
                
                for (int i=0; i<forward.size(); ++i) {
                    forward.at(i)->forwardPropagation();
                }
            }

            virtual void calcPartialDerivative(dtype eta)
            {
                for (int i=0; i<bsize; ++i) {
                    dtype err = back.at(0)->data.at(i) - back.at(1)->data.at(i);
                    back.at(0)->grad.at(i) =   2 * err * grad.at(0) / bsize;
                    back.at(1)->grad.at(i) = - 2 * err * grad.at(0) / bsize;
                }
            }      
    };

    void dumpNode (Node const node1, std::string name)
    {
        std::cout << name << " back size = " << node1.back.size()    << std::endl;
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
