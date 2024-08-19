#include <string>
#include <map>
#include <fstream>
#include "Type.hpp"
#include "CG.hpp"
#include "CGconverter.hpp"

namespace CGC
{
    Converter::Converter (){};

    void Converter::convertAll(CG::Node *top, std::string filename)
    {   
        assert (top->forward.size() == 0);

        int index = 0;
        id = &index;
        p2i = {};

        std::ofstream outputFile(filename, std::ios::out);
        out = &outputFile;
        
        convert(top);
        (*out).close();
    }

    void Converter::convert(CG::Node *node)
    {
        if (p2i.find(node) == p2i.end()) {
            p2i[node] = ++*id;
        } else {
            return;
        }

        toString(node);

      /*for (int i=0; i<node->forward.size(); ++i) {
            convert(node->forward.at(i), id);
        }*/

        for (int i=0; i<node->backward.size(); ++i) {
            convert(node->backward.at(i));
        }
    }

    void Converter::toString(CG::Node *node)
    {   
        assert (p2i.find(node) != p2i.end());

        if (typeid(*node) == typeid(CG::Leaf)) {
            *out << "id " << p2i[node] << std::endl;
            *out << "Node Leaf" << std::endl;
            *out << "data " << node->data.size() << std::endl;
        } else if (typeid(*node) == typeid(CG::Add)) {
            if (p2i.find(node->backward.at(0)) == p2i.end()) {
                convert(node->backward.at(0));
            }
            if (p2i.find(node->backward.at(1)) == p2i.end()) {
                convert(node->backward.at(1));
            }
            *out << "id " << p2i[node] << std::endl;
            *out << "Node Add" << std::endl;
            *out << "back " << p2i[node->backward.at(0)] << " " << p2i[node->backward.at(1)] << std::endl;
        } else if (typeid(*node) == typeid(CG::Sub)) {
            if (p2i.find(node->backward.at(0)) == p2i.end()) {
                convert(node->backward.at(0));
            }
            if (p2i.find(node->backward.at(1)) == p2i.end()) {
                convert(node->backward.at(1));
            }
            *out << "id " << p2i[node] << std::endl;
            *out << "Node Sub" << std::endl;
            *out << "back " << p2i[node->backward.at(0)] << " " << p2i[node->backward.at(1)] << std::endl;
        } else if (typeid(*node) == typeid(CG::Dots)) {
            if (p2i.find(node->backward.at(0)) == p2i.end()) {
                convert(node->backward.at(0));
            }
            if (p2i.find(node->backward.at(1)) == p2i.end()) {
                convert(node->backward.at(1));
            }
            *out << "id " << p2i[node] << std::endl;
            *out << "Node Dots" << std::endl;
            *out << "back " << p2i[node->backward.at(0)] << " " << p2i[node->backward.at(1)] << std::endl;
        } else if (typeid(*node) == typeid(CG::MLE)) {
            if (p2i.find(node->backward.at(0)) == p2i.end()) {
                convert(node->backward.at(0));
            }
            if (p2i.find(node->backward.at(1)) == p2i.end()) {
                convert(node->backward.at(1));
            }
            *out << "id " << p2i[node] << std::endl;
            *out << "Node MLE" << std::endl;
            *out << "back " << p2i[node->backward.at(0)] << " " << p2i[node->backward.at(1)] << std::endl;
        } else if (typeid(*node) == typeid(CG::ReLU)) {
            if (p2i.find(node->backward.at(0)) == p2i.end()) {
                convert(node->backward.at(0));
            }
            *out << "id " << p2i[node] << std::endl;
            *out << "Node ReLU" << std::endl;
            *out << "back " << p2i[node->backward.at(0)] << std::endl;
        } else if (typeid(*node) == typeid(CG::Softmax)) {
            if (p2i.find(node->backward.at(0)) == p2i.end()) {
                convert(node->backward.at(0));
            }
            *out << "id " << p2i[node] << std::endl;
            *out << "Node Softmax" << std::endl;
            *out << "back " << p2i[node->backward.at(0)] << std::endl;
        } else if (typeid(*node) == typeid(CG::Norm2)) {
            if (p2i.find(node->backward.at(0)) == p2i.end()) {
                convert(node->backward.at(0));
            }
            *out << "id " << p2i[node] << std::endl;
            *out << "Node Norm2" << std::endl;
            *out << "back " << p2i[node->backward.at(0)] << std::endl;
        } else if (typeid(*node) == typeid(CG::Affine)) {
            if (p2i.find(node->backward.at(0)) == p2i.end()) {
                convert(node->backward.at(0));
            }
            CG::Affine *aff = dynamic_cast<CG::Affine*>(node);
            assert (aff != nullptr);
            *out << "id " << p2i[aff] << std::endl;
            *out << "Node Affine" << std::endl;
            *out << "back " << p2i[aff->backward.at(0)] << std::endl;
            *out << "bias " << aff->bias << std::endl;
            *out << "Weight " << aff->domsize << " " << aff->data.size() << std::endl;
            for (int i=0; i<aff->Weight.size(); ++i) {
                for (int j=0; j<aff->Weight.at(i).size(); ++j) {
                    if (j != 0) {
                        *out << " ";
                    }
                    *out << aff->Weight.at(i).at(j);
                }
                *out << std::endl;
            }
        } else {
            assert (false);
        }

        *out << std::endl;
    }
}
