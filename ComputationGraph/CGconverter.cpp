#include <fstream>
#include <map>
#include "CG.hpp"
#include "CGconverter.hpp"

namespace CGC
{
    Converter::Converter (){};

    void Converter::convertAll(CG::Node *top, std::string filename)
    {   
        assert (top->forward.size() == 0);

        int id = 0;
        p2i = {};
        i2p = {};

        std::ofstream outputFile(filename, std::ios::out);
        convert(top, &id, &outputFile);
        outputFile.close();
    }

    void Converter::convert(CG::Node *node, int *id, std::ofstream *out)
    {
        if (p2i.find(node) == p2i.end()) {
            p2i[node] = ++*id;
            i2p[*id]  = node;
        } else {
            return;
        }

        toString(node, out);

        for (int i=0; i<node->backward.size(); ++i) {
            convert(node, id, out);
        }
    }

    void Converter::toString(CG::Node *node,  std::ofstream *out)
    {   
        assert (p2i.find(node) != p2i.end());

        *out << "id " << p2i[node] << std::endl;

        if (typeid(*node) == typeid(CG::Leaf)) {
            *out << "Node Leaf" << std::endl;
            *out << "data " << node->data.size() << std::endl;
        } else {
            assert (false);
        }
    }
}
