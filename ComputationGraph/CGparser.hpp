#ifndef CGP_HPP
#define CGP_HPP

#include <string>
#include <map>
#include <fstream>
#include "Type.hpp"
#include "CG.hpp"

namespace CGP
{
    template<typename T> using vec1 = type::vec1<T>;
    template<typename T> using vec2 = type::vec2<T>;
    using dtype = type::dtype;

    class Parser
    {   
        public :
            std::map<int, CG::Node*> i2p;
            std::ifstream *in;
            std::string buffer;

            Parser ();

            CG::Node* parseAll(std::string filename);

            CG::Node* parse();
    };
}

#endif
