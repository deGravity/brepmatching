#include <iostream>
#include <assert.h>

#include <cm_lib.h>

int main(void) {
    auto matching = make_matching(SHORTBOX, LONGBOX, false);
    auto shortbox_types = get_export_id_types(SHORTBOX);
    auto longbox_types = get_export_id_types(LONGBOX);

    
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

    std::cout << matching.json();

    return 0;
}