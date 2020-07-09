//
// Created by LevZ on 6/15/2020.
//

#ifndef TARGETPRACTICE_COMMON_H
#define TARGETPRACTICE_COMMON_H

#include <iostream>
#include <string>
#include <unordered_set>


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

private:
    std::string msg;
};

#endif //TARGETPRACTICE_COMMON_H
