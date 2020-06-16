//
// Created by LevZ on 6/15/2020.
//

#ifndef BLAS_COMMON_H
#define BLAS_COMMON_H

#include <iostream>
#include <string>
#include <unordered_set>


class warning : public std::exception {
public:
    static std::unordered_set<std::string> history;
    warning(const std::string& msg) : msg(msg) { warning::history.emplace(msg);}
    const char* what() { return msg.c_str(); }

    static void warn(const std::string& msg) {
#ifdef DEBUG
      throw warning(msg);
#else
      if (warning::history.count(msg) == 0)
          std::cerr << warning(msg).what() << std::endl;
#endif
    }

private:
    std::string msg;
};

#endif //BLAS_COMMON_H
