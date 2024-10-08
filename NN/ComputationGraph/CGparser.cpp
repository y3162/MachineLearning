#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include "CG.hpp"
#include "CGparser.hpp"

namespace CGP
{
    Parser::Parser (){};

    CG::Node* Parser::parseAll(std::string filename)
    {
        std::ifstream inputFile(filename, std::ios::in);

        in = &inputFile;
        CG::Node *ret;
        while (std::getline(*in,  buffer)) {
            if (buffer.empty()) {
                continue;
            }
            ret = parse();
        }
        (*in).close();

        return ret;
    }

    CG::Node* Parser::parse()
    {
        std::string token;
        int id, id1, id2;
        CG::Node *ret;

        std::stringstream ss(buffer);
        ss >> token;
        assert (token == "id");
        ss >> id;

        *in >> token;
        assert (token == "Node");
        *in >> token;

        if (token == "Leaf1") {
            int size;
            *in >> token;
            assert (token == "data");
            *in >> size;

            ret = new CG::Leaf1(size);
            i2p[id] = ret;
            return ret;
        } else if (token == "Leaf2") {
            int height, width;
            *in >> token;
            assert (token == "data");
            *in >> height >> width;
            ret = new CG::Leaf2(height, width);
            i2p[id] = ret;
            return ret;
        } else if (token == "Concatenation") {
            int size;
            vec1<CG::Node*> nodes;
            *in >> token;
            assert (token == "channel");
            *in >> size;
            *in >> token;
            assert (token == "back");
            nodes.resize(size);
            for (int i=0; i<size; ++i) {
                *in >> id1;
                assert (i2p.find(id1) != i2p.end());
                nodes.at(i) = i2p[id1];
            }
            ret = new CG::Concatenation(nodes);
            i2p[id] = ret;
            return ret;
        } else if (token == "Add") {
            *in >> token;
            assert (token == "back");
            *in >> id1;
            *in >> id2;
            assert (i2p.find(id1) != i2p.end());
            assert (i2p.find(id2) != i2p.end());
            ret = new CG::Add(i2p[id1], i2p[id2]);
            i2p[id] = ret;
            return ret;
        } else if (token == "Sub") {
            *in >> token;
            assert (token == "back");
            *in >> id1;
            *in >> id2;
            assert (i2p.find(id1) != i2p.end());
            assert (i2p.find(id2) != i2p.end());
            ret = new CG::Sub(i2p[id1], i2p[id2]);
            i2p[id] = ret;
            return ret;
        } else if (token == "Dots") {
            *in >> token;
            assert (token == "back");
            *in >> id1;
            *in >> id2;
            assert (i2p.find(id1) != i2p.end());
            assert (i2p.find(id2) != i2p.end());
            ret = new CG::Dots(i2p[id1], i2p[id2]);
            i2p[id] = ret;
            return ret;
        } else if (token == "MSE") {
            *in >> token;
            assert (token == "back");
            *in >> id1;
            *in >> id2;
            assert (i2p.find(id1) != i2p.end());
            assert (i2p.find(id2) != i2p.end());
            ret = new CG::MSE(i2p[id1], i2p[id2]);
            i2p[id] = ret;
            return ret;
        } else if (token == "CEE") {
            *in >> token;
            assert (token == "back");
            *in >> id1;
            *in >> id2;
            assert (i2p.find(id1) != i2p.end());
            assert (i2p.find(id2) != i2p.end());
            ret = new CG::CEE(i2p[id1], i2p[id2]);
            i2p[id] = ret;
            return ret;
        } else if (token == "ReLU") {
            *in >> token;
            assert (token == "back");
            *in >> id1;
            assert (i2p.find(id1) != i2p.end());
            ret = new CG::ReLU(i2p[id1]);
            i2p[id] = ret;
            return ret;
        } else if (token == "Sigmoid") {
            *in >> token;
            assert (token == "back");
            *in >> id1;
            assert (i2p.find(id1) != i2p.end());
            ret = new CG::Sigmoid(i2p[id1]);
            i2p[id] = ret;
            return ret;
        } else if (token == "Tanh") {
            *in >> token;
            assert (token == "back");
            *in >> id1;
            assert (i2p.find(id1) != i2p.end());
            ret = new CG::Tanh(i2p[id1]);
            i2p[id] = ret;
            return ret;
        } else if (token == "Softmax") {
            *in >> token;
            assert (token == "back");
            *in >> id1;
            assert (i2p.find(id1) != i2p.end());
            ret = new CG::Softmax(i2p[id1]);
            i2p[id] = ret;
            return ret;
        } else if (token == "Norm2") {
            *in >> token;
            assert (token == "back");
            *in >> id1;
            assert (i2p.find(id1) != i2p.end());
            ret = new CG::Norm2(i2p[id1]);
            i2p[id] = ret;
            return ret;
        } else if (token == "Affine") {
            int M, N;
            vec2<dtype> w;
            dtype b;

            *in >> token;
            assert (token == "back");
            *in >> id1;
            assert (i2p.find(id1) != i2p.end());
            *in >> token;
            assert (token == "bias");
            *in >> b;
            *in >> token;
            assert (token == "weight");
            *in >> M;
            *in >> N;
            w.resize(M+1);
            for (int i=0; i<=M; ++i) {
                w.at(i).resize(N);
                for (int j=0; j<N; ++j) {
                    *in >> w.at(i).at(j);
                }
            }
            CG::Affine *ret1 = new CG::Affine(i2p[id1], w, b);
            i2p[id] = ret1;
            return ret1;
        } else if (token == "Convolution2d") {
            size_t ch, h, w;
            vec1<CG::Node*> nodes;
            size_t s, pt, pl, kh, kw;
            vec3<dtype> k;
            dtype b;

            *in >> token;
            assert (token == "channel");
            *in >> ch;
            nodes.resize(ch);
            *in >> token;
            assert (token == "back");
            for (int c=0; c<ch; ++c) {
                *in >> id1;
                assert (i2p.find(id1) != i2p.end());
                nodes.at(c) = i2p[id1];
            }
            *in >> token;
            assert (token == "data");
            *in >> h >> w;
            *in >> token;
            assert (token == "stride");
            *in >> s;
            *in >> token;
            assert (token == "padding");
            *in >> pt >> pl;
            *in >> token;
            assert (token == "bias");
            *in >> b;
            *in >> token;
            assert (token == "kernel");
            *in >> kh >> kw;
            k.resize(ch);
            for (int c=0; c<ch; ++c) {
                k.at(c).resize(kh);
                for (int i=0; i<kh; ++i) {
                    k.at(c).at(i).resize(kw);
                    for (int j=0; j<kw; ++j) {
                        *in >> k.at(c).at(i).at(j);
                    }
                }
            }
            CG::Convolution2d *ret1 = new CG::Convolution2d(nodes, k, b, s, pt, pl, h, w);
            i2p[id] = ret1;
            return ret1;
        } else if (token == "MaxPooling2d") {
            size_t h, w;
            size_t s, pt, pl, kh, kw;

            *in >> token;
            assert (token == "data");
            *in >> h >> w;
            *in >> token;
            assert (token == "back");
            *in >> id1;
            assert (i2p.find(id1) != i2p.end());
            *in >> token;
            assert (token == "stride");
            *in >> s;
            *in >> token;
            assert (token == "padding");
            *in >> pt >> pl;
            *in >> token;
            assert (token == "filter");
            *in >> kh >> kw;
            CG::MaxPooling2d *ret1 = new CG::MaxPooling2d(i2p[id1], kh, kw, s, h, w);
            i2p[id] = ret1;
            return ret1;
        } else if (token == "AveragePooling2d") {
            size_t h, w;
            size_t s, pt, pl, kh, kw;

            *in >> token;
            assert (token == "data");
            *in >> h >> w;
            *in >> token;
            assert (token == "back");
            *in >> id1;
            assert (i2p.find(id1) != i2p.end());
            *in >> token;
            assert (token == "stride");
            *in >> s;
            *in >> token;
            assert (token == "padding");
            *in >> pt >> pl;
            *in >> token;
            assert (token == "filter");
            *in >> kh >> kw;
            CG::AveragePooling2d *ret1 = new CG::AveragePooling2d(i2p[id1], kh, kw, s, h, w);
            i2p[id] = ret1;
            return ret1;
        } else {
            assert (false);
        }
    }
}
