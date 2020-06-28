//
// Created by LevZ on 6/28/2020.
//

#ifndef TARGETPRACTICE_GRAPHVIZPRINTER_H
#define TARGETPRACTICE_GRAPHVIZPRINTER_H

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <iostream>

using namespace std;

class GraphvizPrinter {
public:
    void create_dependency(const string& dependee, const string& dependent);

    ostream & print_dot(ostream& os);

    void export_to(const string& format);

    size_t create_node(const string &name, const string &style = "");
private:
    unordered_map<string /*label*/, size_t /*id*/> nodes;
    unordered_map<string /*label*/, string /*style*/> styles;
    unordered_set<string /*node1 -> node2*/ > connections;
};


#endif //TARGETPRACTICE_GRAPHVIZPRINTER_H
