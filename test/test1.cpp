#include "../ComputationGraph/CG.hpp"

int main(void) {
    CG::Leaf l1(1);
    std::vector<double> v1 = {999};
    l1.getInput(v1);

    l1.forwardPropagation();
    l1.backPropagation();
    
    CG::dumpNode(l1, "l1");

    return 0;
}
