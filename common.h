//
// Created by LevZ on 6/15/2020.
//

#ifndef TARGETPRACTICE_COMMON_H
#define TARGETPRACTICE_COMMON_H

#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
#include <sstream>


class warning : public std::exception {
public:
    static std::unordered_set<std::string> history;
    explicit warning(const std::string& msg) : msg(msg) { warning::history.emplace(msg);}
    const char* what() { return msg.c_str(); }

    static void warn(const std::string& msg) {
#ifndef NDEBUG
      throw warning(msg);
#else
      if (warning::history.count(msg) == 0)
          std::cerr << warning(msg).what() << std::endl;
#endif
    }

#define NOT_IMPLEMENTED { throw std::logic_error("Not Implemented."); };
#define NOT_IMPLEMENTED_CRITICAL { static_assert(false, "NOT IMPLEMENTED"); };
#define MARK_FORBIDDEN(expr) expr { throw std::logic_error("Forbidden: '" #expr "'"); };

private:
    std::string msg;
};

template<typename T>
static inline std::string vec2string(const std::vector<T>& v) {
    std::stringstream ss;
    ss << '[';
    for (int i = 0; i < v.size(); ++i){
        if (i != 0)
            ss << ", ";
        ss << v[i];
    }
    ss << ']';
    return ss.str();
}

#endif //TARGETPRACTICE_COMMON_H
