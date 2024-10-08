#ifndef TYPE_HPP
#define TYPE_HPP

namespace type
{
    template<typename T> using vec1 = std::vector<T>;
    template<typename T> using vec2 = vec1<vec1<T>>;
    template<typename T> using vec3 = vec1<vec2<T>>;
    using dtype = double;
    using ttype = unsigned int;
}

#endif
