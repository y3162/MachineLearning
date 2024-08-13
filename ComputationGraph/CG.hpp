#include <cassert>
#include <vector>

namespace CG
{
    template<typename T> using vec1 = std::vector<T>;
    using dtype = double;

    class Node
    {
        public:
            vec1<dtype> data;
            vec1<dtype> grad;
            vec1<Node*> forward;
            vec1<Node*> back;
            int         f_count = 0;
            int         b_count = 0;

            Node (){}

            virtual void ForwardPropagation(){}
            void BackPropagation(dtype eta)
            {
                if (++f_count < forward.size()) {
                    return;
                } else { // When all the backpropagations from the units in the next layer have been completed
                    f_count = 0;
                }

                if (forward.size() == 0) {
                    assert (data.size() == 1);
                    grad.at(0) = 1;
                } else {
                    for (int i=0; i<grad.size(); ++i) {
                        dtype sum = 0;
                        for (int j=0; j<forward.size(); ++j) { // Chain Rule for Partial Derivatives
                            sum += forward.at(j)->grad.at(i);
                        }
                        grad.at(i) = grad.at(i) * sum;
                    }
                }

                for (int i=0; i<back.size(); ++i) {
                    back.at(i)->BackPropagation(eta);
                }
            }
    };

    class Leaf : public Node
    {
        public:
            Leaf (size_t size)
            {
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

            virtual void ForwardPropagation()
            {
                for (int i=0; i<forward.size(); ++i) {
                    forward.at(i)->ForwardPropagation();
                }
            }
    };

    class Plus : public Node
    {   
        public:
            Plus (Node *node1, Node *node2)
            {   
                assert (node1->data.size() == node2->data.size());

                size_t size = node1->data.size();
                data.resize(size);
                grad.resize(size);

                forward.resize(0);

                back.resize(2);
                back.at(0) = node1;
                back.at(1) = node2;

                int fsize1 = node1->forward.size();
                node1->forward.resize(fsize1+1);
                node1->forward.at(fsize1) = this;
                int fsize2 = node2->forward.size();
                node2->forward.resize(fsize2+1);
                node2->forward.at(fsize2) = this;
            };

            virtual void ForwardPropagation()
            {   
                if (++b_count < back.size()) {
                    return;
                } else { // When all the forward passes from the units in the preceding layer have been completed
                    b_count = 0;
                }

                for (int i=0; i<data.size(); ++i) {
                    data.at(i) = back.at(0)->data.at(i) + back.at(1)->data.at(i);
                }
                for (int i=0; i<forward.size(); ++i) {
                    forward.at(i)->ForwardPropagation();
                }

                for (int i=0; i<back.at(0)->grad.size(); ++i) {
                    back.at(0)->grad.at(i) = 1;
                    back.at(1)->grad.at(i) = 1;
                }
            }
    };

    class Dots : public Plus
    {
        public:
            Dots (Node *node1, Node *node2) : Plus (node1, node2){};

            virtual void ForwardPropagation()
            {
                if (++b_count < back.size()) {
                    return;
                } else { // When all the forward passes from the units in the preceding layer have been completed
                    b_count = 0;
                }

                for (int i=0; i<data.size(); ++i) {
                    data.at(i) = back.at(0)->data.at(i) * back.at(1)->data.at(i);
                }
                for (int i=0; i<forward.size(); ++i) {
                    forward.at(i)->ForwardPropagation();
                }
                
                for (int i=0; i<back.at(0)->grad.size(); ++i) {
                    back.at(0)->grad.at(i) = back.at(1)->data.at(i);
                    back.at(1)->grad.at(i) = back.at(0)->data.at(i);
                }
            }
    };
};
