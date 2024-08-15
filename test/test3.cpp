#include "../ComputationGraph/CG.hpp"

int main(void) {
    CG::Leaf l1(4);
    std::vector<double> v1 = {1, 2, 3, 4};
    l1.getInput(v1);

    CG::Leaf l2(4);
    std::vector<double> v2 = {3, 1, 5, 5};
    l2.getInput(v2);

    CG::Dots d1(&l1, &l2);

    l1.forwardPropagation();
    l2.forwardPropagation();
    d1.backPropagation();

    CG::dumpNode(l1, "l1");
    CG::dumpNode(l2, "l2");
    CG::dumpNode(d1, "d1");

    return 0;
}
