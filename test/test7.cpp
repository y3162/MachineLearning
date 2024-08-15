#include "../ComputationGraph/CG.hpp"

int main(void) {
    CG::Leaf l1(2);
    std::vector<double> v1 = {-1, 2};
    l1.getInput(v1);

    CG::Leaf l2(2);
    v1 = {6, 3};
    l2.getInput(v1);

    CG::Leaf l3(2);
    v1 = {10, 1};
    l3.getInput(v1);

    CG::Add a1(&l1, &l2);

    CG::MLE m1(&a1, &l3);

    l1.forwardPropagation();
    l2.forwardPropagation();
    l3.forwardPropagation();
    m1.backPropagation();

    CG::dumpNode(l1, "l1");
    CG::dumpNode(l2, "l2");
    CG::dumpNode(l3, "l3");
    CG::dumpNode(a1, "a1");
    CG::dumpNode(m1, "m1");

    return 0;
}
