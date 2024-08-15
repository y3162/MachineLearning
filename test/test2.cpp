#include "../ComputationGraph/CG.hpp"

int main(void) {
    CG::Leaf l1(1);
    std::vector<double> v1 = {333};
    l1.getInput(v1);

    CG::Leaf l2(1);
    std::vector<double> v2 = {666};
    l2.getInput(v2);

    CG::Add a1(&l1, &l2);

    l1.forwardPropagation();
    l2.forwardPropagation();
    a1.backPropagation();

    CG::dumpNode(l1, "l1");
    CG::dumpNode(l2, "l2");
    CG::dumpNode(a1, "a1");

    return 0;
}
