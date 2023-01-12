#ifndef CM_LIB_INCLUDED
#define CM_LIB_INCLUDED

#include <vector>
#include <string>
#include <tuple>
#include <sstream>
#include <map>

struct Matching {
    std::vector<std::tuple<std::string, std::string>> face_matches;
    std::vector<std::tuple<std::string, std::string>> edge_matches;
    std::vector<std::tuple<std::string, std::string>> vertex_matches;

    std::vector<std::tuple<std::string, std::string>> face_overlaps;
    std::vector<std::tuple<std::string, std::string>> edge_overlaps;
    std::vector<std::tuple<std::string, std::string>> vertex_overlaps;

    std::string json() const {
        std::stringstream ss;
        ss << "{\n";
        for (auto matches : { face_matches, face_overlaps, edge_matches, edge_overlaps, vertex_matches, vertex_overlaps }) {
            for (auto match : matches) {
                ss << "\t\"" << std::get<0>(match) << "\":\"" << std::get<1>(match) << "\",\n";
            }
        }
        ss << "}";
        return ss.str();
    }

};

Matching make_matching(std::string part1, std::string part2, bool exact=false);

std::map<std::string, std::string> get_export_id_types(std::string part);

#endif