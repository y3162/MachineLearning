#include "../ComputationGraph/CG.hpp"

int main(void) {
    CG::Leaf l1(2);
    std::vector<double> v1 = {-1, 2};
    l1.getInput(v1);

    CG::Leaf l2(2);
    std::vector<double> v2 = {6, 3};
    l2.getInput(v2);

    CG::Sub s1(&l1, &l2);

    CG::Affine a1(&s1, {{-1, 1},
                        { 2, 4},
                        {-1, 8}});

    CG::Norm2 n1(&a1);

    l1.forwardPropagation();
    l2.forwardPropagation();
    n1.backPropagation();
    n1.update(0.1);

    CG::dumpNode(n1, "n1");

    l1.forwardPropagation();
    l2.forwardPropagation();
    n1.backPropagation();

    CG::dumpNode(n1, "n1");

    return 0;
}
