#include <iostream>
#include <assert.h>

#include <cm_lib.h>

int main(void) {
    // short and long, and long and longer should have overlaps, but short and longer shouldn't
    auto partA = SMALLCUBE;
    auto partB = SHIFTCUBE;

    auto matching = make_matching(partA, partB, false);

    auto shortbox_types = get_export_id_types(partA);
    auto longbox_types = get_export_id_types(partB);

    
    std::cout << "Face Matches:\n";
    for (auto m : matching.face_matches) {
        auto& id1 = std::get<0>(m);
        auto& id2 = std::get<1>(m);
        std::cout << id1 << "(" << shortbox_types[id1] << ")" << " , " << id2<< "(" << longbox_types[id2] << ")" << "\n";
    }

    std::cout << "Edge Matches:\n";
    for (auto m : matching.edge_matches) {
        auto& id1 = std::get<0>(m);
        auto& id2 = std::get<1>(m);
        std::cout << id1 << "(" << shortbox_types[id1] << ")" << " , " << id2 << "(" << longbox_types[id2] << ")" << "\n";
    }

    std::cout << "Vertex Matches:\n";
    for (auto m : matching.vertex_matches) {
        auto& id1 = std::get<0>(m);
        auto& id2 = std::get<1>(m);
        std::cout << id1 << "(" << shortbox_types[id1] << ")" << " , " << id2 << "(" << longbox_types[id2] << ")" << "\n";
    }

    std::cout << "Face Overlaps:\n";
    for (auto m : matching.face_overlaps) {
        auto& id1 = std::get<0>(m);
        auto& id2 = std::get<1>(m);
        std::cout << id1 << "(" << shortbox_types[id1] << ")" << " , " << id2 << "(" << longbox_types[id2] << ")" << "\n";
    }

    std::cout << "Face Overlap Pct Large\n";
    for (auto& pct : matching.larger_face_overlap_percentages) {
        std::cout << pct << ", ";
    }
    std::cout << "\n";


    std::cout << "Face Overlap Pct Small\n";
    for (auto& pct : matching.smaller_face_overlap_percentages) {
        std::cout << pct << ", ";
    }
    std::cout << "\n";

    std::cout << "Edge Overlaps:\n";
    for (auto m : matching.edge_overlaps) {
        auto& id1 = std::get<0>(m);
        auto& id2 = std::get<1>(m);
        std::cout << id1 << "(" << shortbox_types[id1] << ")" << " , " << id2 << "(" << longbox_types[id2] << ")" << "\n";
    }

    std::cout << "Edge Overlap Pct Large\n";
    for (auto& pct : matching.larger_face_overlap_percentages) {
        std::cout << pct << ", ";
    }
    std::cout << "\n";


    std::cout << "Edge Overlap Pct Small\n";
    for (auto& pct : matching.smaller_face_overlap_percentages) {
        std::cout << pct << ", ";
    }
    std::cout << "\n";

    std::cout << "JSON Representation:\n";
    std::cout << matching.json();

    return 0;
}