#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cm_lib.h>
#include <map>
#include <string>

Matching match_parts(std::string p1, std::string p2, bool exact=false) {
	auto matching = make_matching(p1, p2, exact);
	return matching;
}

std::map<std::string, std::string> match_parts_dict(std::string p1, std::string p2, bool exact = false) {
	auto matching = make_matching(p1, p2, exact);
	std::map<std::string, std::string> matching_dict;
	for (auto matches : { matching.face_matches, matching.edge_matches, matching.vertex_matches }) {
		for (auto match : matches) {
			auto m1 = std::get<0>(match);
			auto m2 = std::get<1>(match);
			matching_dict[m1] = m2;
		}
	}
	return matching_dict;
}

PYBIND11_MODULE(coincidence_matching, m) {
	pybind11::class_<Matching>(m, "Matching")
		.def_readwrite("face_matches", &Matching::face_matches)
		.def_readwrite("edge_matches", &Matching::edge_matches)
		.def_readwrite("vertex_matches", &Matching::vertex_matches)
		.def("__repr__",
			[](const Matching& matching) {
				return matching.json();
			});
	m.def("match_parts", &match_parts);
	m.def("match_parts_dict", &match_parts_dict);
	m.def("get_export_id_types", &get_export_id_types);
}