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

        if (typeid(*node) == typeid(CG::Leaf1)) {
            *out << "id " << p2i[node] << std::endl;
            *out << "Node Leaf1" << std::endl;
            *out << "data " << node->data.size() << std::endl;
        } else if (typeid(*node) == typeid(CG::Leaf2)) {
            *out << "id " << p2i[node] << std::endl;
            *out << "Node Leaf2" << std::endl;
            *out << "data " << node->height << " " << node->width << std::endl;
        } else if (typeid(*node) == typeid(CG::Concatenation)) {
            for (int i=0; i<node->backward.size(); ++i) {
                if (p2i.find(node->backward.at(i)) == p2i.end()) {
                    convert(node->backward.at(i));
                }
            }
            CG::Concatenation*conv = dynamic_cast<CG::Concatenation*>(node);
            assert (conv != nullptr);
            *out << "id " << p2i[conv] << std::endl;
            *out << "Node Concatenation" << std::endl;
            *out << "channel " << node->backward.size() << std::endl;
            *out << "back";
            for (int i=0; i<node->backward.size(); ++i) {
                *out << " " << p2i[node->backward.at(i)];
            }
            *out << std::endl;
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
        } else if (typeid(*node) == typeid(CG::MSE)) {
            if (p2i.find(node->backward.at(0)) == p2i.end()) {
                convert(node->backward.at(0));
            }
            if (p2i.find(node->backward.at(1)) == p2i.end()) {
                convert(node->backward.at(1));
            }
            *out << "id " << p2i[node] << std::endl;
            *out << "Node MSE" << std::endl;
            *out << "back " << p2i[node->backward.at(0)] << " " << p2i[node->backward.at(1)] << std::endl;
        } else if (typeid(*node) == typeid(CG::CEE)) {
            if (p2i.find(node->backward.at(0)) == p2i.end()) {
                convert(node->backward.at(0));
            }
            if (p2i.find(node->backward.at(1)) == p2i.end()) {
                convert(node->backward.at(1));
            }
            *out << "id " << p2i[node] << std::endl;
            *out << "Node CEE" << std::endl;
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
            *out << "weight " << aff->domsize << " " << aff->data.size() << std::endl;
            for (int i=0; i<aff->weight.size(); ++i) {
                for (int j=0; j<aff->weight.at(i).size(); ++j) {
                    if (j != 0) {
                        *out << " ";
                    }
                    *out << aff->weight.at(i).at(j);
                }
                *out << std::endl;
            }
        } else if (typeid(*node) == typeid(CG::Convolution2d)) {
            if (p2i.find(node->backward.at(0)) == p2i.end()) {
                convert(node->backward.at(0));
            }
            CG::Convolution2d*conv = dynamic_cast<CG::Convolution2d*>(node);
            assert (conv != nullptr);
            *out << "id " << p2i[conv] << std::endl;
            *out << "Node Convolution2d" << std::endl;
            *out << "channel " << conv->backward.size() << std::endl;
            *out << "back";
            for (int i=0; i<node->backward.size(); ++i) {
                *out << " " << p2i[node->backward.at(i)];
            }
            *out << std::endl;
            *out << "data " << conv->height << " " << conv->width << std::endl;
            *out << "stride " << conv->sw << std::endl;
            *out << "padding " << conv->pt << " " << conv->pt << std::endl;
            *out << "bias " << conv->bias << std::endl;
            *out << "kernel " << conv->kheight << " " << conv->kwidth << std::endl;
            for (int c=0; c<conv->backward.size(); ++c) {
                for (int i=0; i<conv->kernel.at(c).size(); ++i) {
                    for (int j=0; j<conv->kernel.at(c).at(i).size(); ++j) {
                        if (j != 0) {
                            *out << " ";
                        }
                        *out << conv->kernel.at(c).at(i).at(j);
                    }
                    *out << std::endl;
                }
                //*out << std::endl;
            }
        } else if (typeid(*node) == typeid(CG::MaxPooling2d)) {
            if (p2i.find(node->backward.at(0)) == p2i.end()) {
                convert(node->backward.at(0));
            }
            CG::MaxPooling2d *maxp = dynamic_cast<CG::MaxPooling2d*>(node);
            assert (maxp != nullptr);
            *out << "id " << p2i[maxp] << std::endl;
            *out << "Node MaxPooling2d" << std::endl;
            *out << "data " << maxp->height << " " << maxp->width << std::endl;
            *out << "back " << p2i[maxp->backward.at(0)] << std::endl;
            *out << "stride " << maxp->sw << std::endl;
            *out << "padding " << maxp->pt << " " << maxp->pt << std::endl;
            *out << "filter " << maxp->kheight << " " << maxp->kwidth << std::endl;
        } else if (typeid(*node) == typeid(CG::AveragePooling2d)) {
            if (p2i.find(node->backward.at(0)) == p2i.end()) {
                convert(node->backward.at(0));
            }
            CG::AveragePooling2d *maxp = dynamic_cast<CG::AveragePooling2d*>(node);
            assert (maxp != nullptr);
            *out << "id " << p2i[maxp] << std::endl;
            *out << "Node AveragePooling2d" << std::endl;
            *out << "data " << maxp->height << " " << maxp->width << std::endl;
            *out << "back " << p2i[maxp->backward.at(0)] << std::endl;
            *out << "stride " << maxp->sw << std::endl;
            *out << "padding " << maxp->pt << " " << maxp->pt << std::endl;
            *out << "filter " << maxp->kheight << " " << maxp->kwidth << std::endl;
        } else {
            assert (false);
        }

        *out << std::endl;
    }
}
