#include "../ComputationGraph/CG.hpp"

int main(void) {
    CG::Leaf l1(4);
    std::vector<double> v1 = {7, 6, 5, 4};
    l1.getInput(v1);

    CG::Leaf l2(4);
    std::vector<double> v2 = {0, 7, 6, -3};
    l2.getInput(v2);

    CG::Sub s1(&l1, &l2);

    CG::Norm2 n1(&s1);

    l1.forwardPropagation();
    l2.forwardPropagation();
    n1.backPropagation();
    n1.update(1);

    CG::dumpNode(l1, "l1");
    CG::dumpNode(l2, "l2");
    CG::dumpNode(s1, "s1");
    CG::dumpNode(n1, "n1");

    return 0;
}
