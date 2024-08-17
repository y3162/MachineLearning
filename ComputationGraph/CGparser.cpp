#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include "CG.hpp"
#include "CGparser.hpp"

namespace CGP
{
    template<typename T> using vec1 = CG::vec1<T>;
    template<typename T> using vec2 = CG::vec2<T>;
    using dtype = CG::dtype;

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

        if (token == "Leaf") {
            int size;
            *in >> token;
            assert (token == "data");
            *in >> size;

            ret = new CG::Leaf(size);
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
        } else if (token == "MLE") {
            *in >> token;
            assert (token == "back");
            *in >> id1;
            *in >> id2;
            assert (i2p.find(id1) != i2p.end());
            assert (i2p.find(id2) != i2p.end());
            ret = new CG::MLE(i2p[id1], i2p[id2]);
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
            assert (token == "Weight");
            *in >> M;
            *in >> N;
            w.resize(M+1);
            for (int i=0; i<=M; ++i) {
                w.at(i).resize(N);
                for (int j=0; j<N; ++j) {
                    *in >> w.at(i).at(j);
                }
            }
            CG::Affine *ret1 = new CG::Affine(i2p[id1], w);
            ret1->bias = b;
            i2p[id] = ret1;
            return ret1;
        } else {
            assert (false);
        }
    }
}
