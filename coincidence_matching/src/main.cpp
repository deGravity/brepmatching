#include <parasolid.h>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <assert.h>
#include <iostream>

bool is_psbody(int id) {
    PK_ERROR_t err = PK_ERROR_no_errors;
    PK_CLASS_t entity_class;
    err = PK_ENTITY_ask_class(id, &entity_class);
    return (err == 0) && (entity_class == PK_CLASS_body);
}

std::vector<int> read_xt(std::string path) {
    ensure_parasolid_session();
    PK_PART_receive_o_t receive_opts;
    PK_PART_receive_o_m(receive_opts);
    receive_opts.transmit_format = PK_transmit_format_text_c;
    int n_parts = 0;
    PK_PART_t* parts = NULL;
    PK_ERROR_t err = PK_ERROR_no_errors;
    err = PK_PART_receive(path.c_str(), &receive_opts, &n_parts, &parts);
    assert(err == PK_ERROR_no_errors); // PK_PART_receive

    std::vector<int> ps_bodies;
    
    if (err == 0) {
        for (int i = 0; i < n_parts; ++i) {
            if (is_psbody(parts[i])) {
                ps_bodies.push_back(parts[i]);
            }
        }
    }
    PK_MEMORY_free(parts); // Do I need to do this, or is this causing segfaults?
    return ps_bodies;
}

std::map<int, std::string> read_attributes(int body_id) {

    PK_ERROR_t err = PK_ERROR_no_errors;

    // Get Names for parts generated in OnShape
    std::map<int, std::string> topo_to_id;
    int num_attdefs;
    PK_ATTDEF_t* attdefs;

    err = PK_PART_ask_all_attdefs(body_id, &num_attdefs, &attdefs);
    assert(err == PK_ERROR_no_errors); // PK_PART_as_all_attdefs
    for (int i = 0; i < num_attdefs; ++i) {
        PK_ATTDEF_t attdef = attdefs[i];
        PK_ATTDEF_sf_t def;
        PK_ATTDEF_ask(attdef, &def);

        if (strcmp(def.name, "BTI/ExportId") == 0) {
            int num_attribs;
            PK_ATTRIB_t* attribs;
            PK_PART_ask_all_attribs(body_id, attdef, &num_attribs, &attribs);
            assert(err == PK_ERROR_no_errors); // PK_PART_ask_all_attribs

            for (int j = 0; j < num_attribs; ++j) {
                PK_ATTRIB_t attrib = attribs[j];
                char* buffer;
                PK_ATTRIB_ask_string(attrib, 0, &buffer);
                assert(err == PK_ERROR_no_errors); // PK_ATTRIB_ask_string
                PK_ENTITY_t owner;
                PK_ATTRIB_ask_owner(attrib, &owner);
                assert(err == PK_ERROR_no_errors); // PK_ATTRIB_ask_owner
                std::string id(buffer);
                topo_to_id[owner] = id;
                delete buffer;
            }
        }
    }
    return topo_to_id;
}

struct BRepTopology
{
    std::vector<int> faces;
    std::vector<int> edges;
    std::vector<int> vertices;
};

BRepTopology read_topology(int body_id) {

    BRepTopology topology;

    PK_ERROR_t err = PK_ERROR_no_errors;

    PK_BODY_ask_topology_o_t ask_topology_options;
    PK_BODY_ask_topology_o_m(ask_topology_options);
    ask_topology_options.want_fins = PK_LOGICAL_false;
    PK_TOPOL_t* topols;
    PK_CLASS_t* classes;
    int n_topols;
    int n_relations;
    int* parents;
    int* children;
    PK_TOPOL_sense_t* senses;
    err = PK_BODY_ask_topology(
        body_id,
        &ask_topology_options,
        &n_topols,
        &topols,
        &classes,
        &n_relations,
        &parents,
        &children,
        &senses);
    assert(err == PK_ERROR_no_errors); // PK_BODY_ask_topology

    for (int i = 0; i < n_topols; ++i) {
        switch (classes[i]) {
        case PK_CLASS_face:
            topology.faces.push_back(topols[i]);
            break;
        case PK_CLASS_edge:
            topology.edges.push_back(topols[i]);
            break;
        case PK_CLASS_vertex:
            topology.vertices.push_back(topols[i]);
            break;
        default:
            break;
        }
    }
    return topology;
}

bool points_coincident(double* p1, double* p2) {
    double d2 = (p1[0] - p2[0])*(p1[0] - p2[0]) + 
    (p1[1] - p2[1])*(p1[1] - p2[1]) + 
    (p1[2] - p2[2])*(p1[2] - p2[2]);
    return d2 <= 1e-16;
}

struct Matching {
    std::vector<std::tuple<std::string, std::string>> face_matches;
    std::vector<std::tuple<std::string, std::string>> edge_matches;
    std::vector<std::tuple<std::string, std::string>> vertex_matches;
};

Matching make_matching(std::string part1, std::string part2, bool exact=false) {

    auto bodies1 = read_xt(part1);
    auto bodies2 = read_xt(part2);

    assert(bodies1.size() == 1);
    assert(bodies2.size() == 1);

    auto p1_topo_map = read_attributes(bodies1[0]);
    auto p2_topo_map = read_attributes(bodies2[0]);

    auto p1_topo = read_topology(bodies1[0]);
    auto p2_topo = read_topology(bodies2[0]);

    PK_ERROR_t err = PK_ERROR_no_errors;

    std::vector<std::tuple<std::string, std::string>> face_matches;
    std::vector<std::tuple<std::string, std::string>> edge_matches;
    std::vector<std::tuple<std::string, std::string>> vertex_matches;

    // Match Faces
    for (auto p1_face : p1_topo.faces) {
        for (auto p2_face : p2_topo.faces) {

            double tol = 1e-8;
            PK_FACE_is_coincident_o_t opts;
            PK_FACE_is_coincident_o_m(opts);
            PK_FACE_coi_t coi;
            PK_VECTOR_t point;

            err = PK_FACE_is_coincident(p1_face, p2_face, tol, &opts, &coi, &point);
            assert(err == PK_ERROR_no_errors); // PK_FACE_is_coincident

            /*
            if (coi == PK_FACE_coi_yes_c || coi == PK_FACE_coi_yes_reversed_c) {
                auto p1_face_id_it = p1_topo_map.find(p1_face);
                auto p2_face_id_it = p2_topo_map.find(p2_face);
                assert(p1_face_id_it != p1_topo_map.end());
                assert(p2_face_id_it != p2_topo_map.end());
                face_matches.push_back(std::make_tuple(p1_face_id_it->second, p2_face_id_it->second));
                break;
            }
            */

            
            if (!exact) {
                PK_SURF_t p1_surf, p2_surf;
                err = PK_FACE_ask_surf(p1_face, &p1_surf);
                assert(err == PK_ERROR_no_errors); // PK_FACE_ask_surf
                err = PK_FACE_ask_surf(p2_face, &p2_surf);
                assert(err == PK_ERROR_no_errors); // PK_FACE_ask_surf

                PK_CLASS_t p1_surf_class, p2_surf_class;

                PK_ENTITY_ask_class(p1_surf, &p1_surf_class);
                assert(err == PK_ERROR_no_errors); // PK_ENTITY_ask_class
                PK_ENTITY_ask_class(p2_surf, &p2_surf_class);
                assert(err == PK_ERROR_no_errors); // PK_ENTITY_ask_class

                if (p1_surf_class != p2_surf_class) continue;
                if ((p1_surf_class == PK_CLASS_plane) || (p1_surf_class == PK_CLASS_cyl) || (p1_surf_class == PK_CLASS_cone) || (p1_surf_class == PK_CLASS_torus) || (p1_surf_class == PK_CLASS_sphere)) {
                    
                    PK_LOGICAL_t surfs_coincident;
                    err = PK_GEOM_is_coincident(p1_surf, p2_surf, &surfs_coincident);
                    assert(err == PK_ERROR_no_errors); // PK_GEOM_is_coincident

                    if (!surfs_coincident) continue;

                    PK_FACE_make_sheet_bodies_o_t make_sheet_opts;
                    PK_FACE_make_sheet_bodies_o_m(make_sheet_opts);
                    int n_bodies1;
                    PK_BODY_t* bodies1;
                    PK_TOPOL_track_r_t tracking1;
                    err = PK_FACE_make_sheet_bodies(1, &p1_face, &make_sheet_opts, &n_bodies1, &bodies1, &tracking1);
                    assert(err == PK_ERROR_no_errors);

                    int n_bodies2;
                    PK_BODY_t* bodies2;
                    PK_TOPOL_track_r_t tracking2;
                    err = PK_FACE_make_sheet_bodies(1, &p1_face, &make_sheet_opts, &n_bodies2, &bodies2, &tracking2);
                    assert(err == PK_ERROR_no_errors);

                    assert(n_bodies1 == 1);
                    assert(n_bodies2 == 1);

                    PK_BODY_boolean_o_t bool_opts;
                    PK_BODY_boolean_o_m(bool_opts);
                    
                    bool_opts.function = PK_boolean_intersect_c;

                    PK_TOPOL_track_r_t bool_tracking;
                    PK_boolean_r_t bool_results;

                    PK_BODY_boolean_2(n_bodies1, 1, &n_bodies2, &bool_opts, &bool_tracking, &bool_results);
                    assert(err == PK_ERROR_no_errors); // PK_BODY_boolean_2

                    PK_TOPOL_eval_mass_props_o_t mass_prop_opts;
                    PK_TOPOL_eval_mass_props_o_m(mass_prop_opts);
                    
                    assert(bool_results.n_bodies == 1);

                    double sheet1_amount, sheet2_amount, intersection_amount, mass, c_of_g[3], m_of_i[9], periphery;

                    err = PK_TOPOL_eval_mass_props(1, bool_results.bodies, 0.999, &mass_prop_opts, &intersection_amount, &mass, c_of_g, m_of_i, &periphery);
                    assert(err == PK_ERROR_no_errors); // PK_TOPOL_eval_mass_props

                    err = PK_TOPOL_eval_mass_props(1, &p1_face, 0.999, &mass_prop_opts, &intersection_amount, &mass, c_of_g, m_of_i, &periphery);
                    assert(err == PK_ERROR_no_errors); // PK_TOPOL_eval_mass_props
                    err = PK_TOPOL_eval_mass_props(1, &p2_face, 0.999, &mass_prop_opts, &intersection_amount, &mass, c_of_g, m_of_i, &periphery);
                    assert(err == PK_ERROR_no_errors); // PK_TOPOL_eval_mass_props

                    double original_size = sheet1_amount < sheet2_amount ? sheet1_amount : sheet2_amount;

                    if (intersection_amount >= .8 * original_size) {
                        auto p1_face_id_it = p1_topo_map.find(p1_face);
                        auto p2_face_id_it = p2_topo_map.find(p2_face);
                        assert(p1_face_id_it != p1_topo_map.end());
                        assert(p2_face_id_it != p2_topo_map.end());
                        face_matches.push_back(std::make_tuple(p1_face_id_it->second, p2_face_id_it->second));
                        break;
                    }

                    PK_MEMORY_free(bodies1);
                    PK_MEMORY_free(bodies2);
                }
            }
        }
    }

    // Match Edges
    for (auto p1_edge : p1_topo.edges) {
        for (auto p2_edge : p2_topo.edges) {
            
            PK_ERROR_t err = PK_ERROR_no_errors;
            
            PK_CURVE_t curve1;
            PK_CLASS_t curve_class1;
            PK_VECTOR_t ends1[2];
            PK_INTERVAL_t t_int1;
            PK_LOGICAL_t sense1;

            PK_CURVE_t curve2;
            PK_CLASS_t curve_class2;
            PK_VECTOR_t ends2[2];
            PK_INTERVAL_t t_int2;
            PK_LOGICAL_t sense2;

            err = PK_EDGE_ask_geometry(
                p1_edge, PK_LOGICAL_true /*yes interval*/, &curve1, &curve_class1, ends1, &t_int1, &sense1);
            assert(err == PK_ERROR_no_errors); // PK_EDGE_ask_geometry

            err = PK_EDGE_ask_geometry(
                p2_edge, PK_LOGICAL_true /*yes interval*/, &curve2, &curve_class2, ends2, &t_int2, &sense2);
            assert(err == PK_ERROR_no_errors); // PK_EDGE_ask_geometry
            
            PK_LOGICAL_t is_coincident;
            err = PK_GEOM_is_coincident(curve1, curve2, &is_coincident);
            assert(err == PK_ERROR_no_errors);

            if (is_coincident) {
                if ( 
                    (points_coincident(ends1[0].coord, ends2[0].coord) && points_coincident(ends1[1].coord, ends2[1].coord)) ||
                    (points_coincident(ends1[0].coord, ends2[1].coord) && points_coincident(ends1[1].coord, ends2[0].coord))
                ) {
                    auto p1_edge_id_it = p1_topo_map.find(p1_edge);
                    auto p2_edge_id_it = p2_topo_map.find(p2_edge);
                    assert(p1_edge_id_it != p1_topo_map.end());
                    assert(p2_edge_id_it != p2_topo_map.end());
                    edge_matches.push_back(std::make_tuple(p1_edge_id_it->second, p2_edge_id_it->second));
                }
            }
        }
    }

    for (auto v1 : p1_topo.vertices) {
        for (auto v2: p2_topo.vertices) {
            PK_POINT_t p1, p2;
            
            err = PK_VERTEX_ask_point(v1, &p1);
            assert(err == PK_ERROR_no_errors); // PK_VERTEX_ask_point
            err = PK_VERTEX_ask_point(v2, &p2);
            assert(err == PK_ERROR_no_errors); // PK_VERTEX_ask_point
            
            PK_LOGICAL_t is_coincident;
            err = PK_GEOM_is_coincident(p1, p2, &is_coincident);
            assert(err == PK_ERROR_no_errors); // PK_GEOM_is_coincident
            
            if (is_coincident) {
                auto p1_vert_id_it = p1_topo_map.find(v1);
                auto p2_vert_id_it = p2_topo_map.find(v2);
                assert(p1_vert_id_it != p1_topo_map.end());
                assert(p2_vert_id_it != p2_topo_map.end());
                vertex_matches.push_back(std::make_tuple(p1_vert_id_it->second, p2_vert_id_it->second));
            }
        }
    }

    Matching m;
    m.face_matches = face_matches;
    m.edge_matches = edge_matches;
    m.vertex_matches = vertex_matches;

    return m;
}

int main(void) {
    auto matching = make_matching(PART1, PART2);

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

    return 0;
}