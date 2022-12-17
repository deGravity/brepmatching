#include <iostream>

#include <cm_lib.h>

int main(void) {
    auto matching = make_matching(PART1, PART2, true);

    /*
    std::cout << "Face Matches:\n";
    for (auto m : matching.face_matches) {
        std::cout << std::get<0>(m) << " , " << std::get<1>(m) << "\n";
    }

    std::cout << "Edge Matches:\n";
    for (auto m : matching.edge_matches) {
        std::cout << std::get<0>(m) << " , " << std::get<1>(m) << "\n";
    }

    std::cout << "Vertex Matches:\n";
    for (auto m : matching.vertex_matches) {
        std::cout << std::get<0>(m) << " , " << std::get<1>(m) << "\n";
    }
    */

    std::cout << matching.json();

    return 0;
}