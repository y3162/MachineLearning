#ifndef CGC_HPP
#define CGC_HPP

#include <string>
#include <map>
#include <fstream>
#include "CG.hpp"

namespace CGC
{
    template<typename T> using vec1 = CG::vec1<T>;
    template<typename T> using vec2 = CG::vec2<T>;
    using dtype = CG::dtype;

    class Converter
    {
        public :
            int *id;
            std::map<CG::Node*, int> p2i;
            std::ofstream *out;

            Converter ();

            void convertAll(CG::Node *top, std::string filename);

            void convert(CG::Node *node);

            void toString(CG::Node *node);
    };
}

#endif
