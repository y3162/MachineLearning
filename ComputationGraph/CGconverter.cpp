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

        std::ofstream outputFile(filename, std::ios::out);
        convert(top, &id, &outputFile);
        outputFile.close();
    }

    void Converter::convert(CG::Node *node, int *id, std::ofstream *out)
    {
        if (p2i.find(node) == p2i.end()) {
            p2i[node] = ++*id;
        } else {
            return;
        }

        toString(node, id, out);

      /*for (int i=0; i<node->forward.size(); ++i) {
            convert(node->forward.at(i), id, out);
        }*/

        for (int i=0; i<node->backward.size(); ++i) {
            convert(node->backward.at(i), id, out);
        }
    }

    void Converter::toString(CG::Node *node, int *id, std::ofstream *out)
    {   
        assert (p2i.find(node) != p2i.end());

        if (typeid(*node) == typeid(CG::Leaf)) {
            *out << "id " << p2i[node] << std::endl;
            *out << "Node Leaf" << std::endl;
            *out << "data " << node->data.size() << std::endl;
        } else if (typeid(*node) == typeid(CG::Add)) {
            if (p2i.find(node->backward.at(0)) == p2i.end()) {
                convert(node->backward.at(0), id, out);
            }
            if (p2i.find(node->backward.at(1)) == p2i.end()) {
                convert(node->backward.at(1), id, out);
            }
            *out << "id " << p2i[node] << std::endl;
            *out << "Node Add" << std::endl;
            *out << "back " << p2i[node->backward.at(0)] << " " << p2i[node->backward.at(1)] << std::endl;
        } else if (typeid(*node) == typeid(CG::Sub)) {
            if (p2i.find(node->backward.at(0)) == p2i.end()) {
                convert(node->backward.at(0), id, out);
            }
            if (p2i.find(node->backward.at(1)) == p2i.end()) {
                convert(node->backward.at(1), id, out);
            }
            *out << "id " << p2i[node] << std::endl;
            *out << "Node Sub" << std::endl;
            *out << "back " << p2i[node->backward.at(0)] << " " << p2i[node->backward.at(1)] << std::endl;
        } else {
            assert (false);
        }

        *out << std::endl;
    }
}
