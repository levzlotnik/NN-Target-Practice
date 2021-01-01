//
// Created by LevZ on 6/28/2020.
//

#include "GraphvizPrinter.h"
#include <cctype>
#include <cstdlib>
#include <fstream>

const unordered_set<string> GraphvizPrinter::available_formats = { "svg", "png", "jpg" };


void GraphvizPrinter::create_dependency(const string& dependee, const string& dependent) {
    auto dependee_id = to_string(create_node(dependee));
    auto dependent_id = to_string(create_node(dependent));
    string connection = "gvzpv_" + dependent_id + " -> " + "gvzpv_" + dependee_id;
    connections.insert(connection);
}

size_t GraphvizPrinter::create_node(const string &name, const string &style) {
    if (name.empty())
        throw invalid_argument("Empty string is not allowed.");
    if (!isalpha(name[0]))
        throw invalid_argument("Only names begining in letters are allowed.");
    auto it = nodes.find(name);
    if (it != nodes.end())
        return it->second;
    styles[name] = style;
    return nodes[name] = nodes.size();
}

ostream & GraphvizPrinter::print_dot(ostream &os) {
    os << "digraph g{" << endl;
    // First pass: Define all nodes:
    for (const auto& [label, id]: nodes)
        os << "    gvzpv_" << id << " [label=\"" << label << "\" " << styles[label] << " ]" << endl;
    // Second pass: Define connections:
    for (const auto& conn: connections)
        os << "    " << conn << endl;
    os << "}" << endl;
    return os;
}

void GraphvizPrinter::export_to(const string &format) {
    if (available_formats.count(format) < 1)
        throw invalid_argument("Unavailable format: \"" + format + "\".");
    auto filename = "graph.dot";
    ofstream fos(filename);
    print_dot(fos);
    fos.close();
    auto cmdline = "dot -T" + format + " " + filename + " -ograph." + format;
    int error_code = std::system(cmdline.c_str());
    if (error_code)
        throw runtime_error("Command \"" + cmdline + "\" returned error code " + to_string(error_code));
    std::remove(filename);
}
