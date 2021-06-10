//
// Created by LevZ on 6/28/2020.
//

#ifndef TARGETPRACTICE_GRAPHVIZPRINTER_H
#define TARGETPRACTICE_GRAPHVIZPRINTER_H

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <iostream>

class GraphvizPrinter {
public:
    void create_dependency(const std::string& dependee, const std::string& dependent);

    std::ostream& print_dot(std::ostream& os);

    void export_to(const std::string& filename);

    size_t create_node(const std::string& name, const std::string& style = "");
private:
    std::unordered_map<std::string /*label*/, size_t /*id*/> nodes;
    std::unordered_map<std::string /*label*/, std::string /*style*/> styles;
    std::unordered_set<std::string /*node1 -> node2*/ > connections;
    static const std::unordered_set<std::string> available_formats;
};


#endif //TARGETPRACTICE_GRAPHVIZPRINTER_H
