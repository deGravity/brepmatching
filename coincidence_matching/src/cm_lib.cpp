#include <parasolid.h>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <assert.h>
#include <iostream>
#include <cm_lib.h>

const double COINCIDENCE_TOL = 1e-5;
const double POINT_EPSILON_2 = 1e-10;
const double MASS_PROP_TOL = 0.999;
const double BOOL_MAX_TOL = 1e-5;
const PK_boolean_match_style_t BOOL_MATCH_STYLE = PK_boolean_match_style_auto_c;

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

std::map<std::string, std::string> get_export_id_types(std::string part) {
    PK_ERROR_code_t err = PK_ERROR_no_errors;
    auto bodies = read_xt(part);
    assert(bodies.size() == 1); // Only one body in the file
    std::map<std::string, std::string> id_to_type;
    auto export_ids = read_attributes(bodies[0]);
    for (auto const& x : export_ids) {
        PK_CLASS_t entity_class;
        err = PK_ENTITY_ask_class(x.first, &entity_class);
        assert(err == PK_ERROR_no_errors); // PK_ENTITY_ask_class
		switch (entity_class) {
		case PK_CLASS_null:
			id_to_type[x.second] = "PK_CLASS_null";
			break;
		case PK_CLASS_class:
			id_to_type[x.second] = "PK_CLASS_class";
			break;
		case PK_CLASS_entity:
			id_to_type[x.second] = "PK_CLASS_entity";
			break;
		case PK_CLASS_primitive:
			id_to_type[x.second] = "PK_CLASS_primitive";
			break;
		case PK_CLASS_error:
			id_to_type[x.second] = "PK_CLASS_error";
			break;
		case PK_CLASS_session:
			id_to_type[x.second] = "PK_CLASS_session";
			break;
		case PK_CLASS_memory:
			id_to_type[x.second] = "PK_CLASS_memory";
			break;
		case PK_CLASS_mark:
			id_to_type[x.second] = "PK_CLASS_mark";
			break;
		case PK_CLASS_pmark:
			id_to_type[x.second] = "PK_CLASS_pmark";
			break;
		case PK_CLASS_partition:
			id_to_type[x.second] = "PK_CLASS_partition";
			break;
		case PK_CLASS_bb:
			id_to_type[x.second] = "PK_CLASS_bb";
			break;
		case PK_CLASS_int:
			id_to_type[x.second] = "PK_CLASS_int";
			break;
		case PK_CLASS_double:
			id_to_type[x.second] = "PK_CLASS_double";
			break;
		case PK_CLASS_char:
			id_to_type[x.second] = "PK_CLASS_char";
			break;
		case PK_CLASS_string:
			id_to_type[x.second] = "PK_CLASS_string";
			break;
		case PK_CLASS_logical:
			id_to_type[x.second] = "PK_CLASS_logical";
			break;
		case PK_CLASS_vector:
			id_to_type[x.second] = "PK_CLASS_vector";
			break;
		case PK_CLASS_interval:
			id_to_type[x.second] = "PK_CLASS_interval";
			break;
		case PK_CLASS_box:
			id_to_type[x.second] = "PK_CLASS_box";
			break;
		case PK_CLASS_uvbox:
			id_to_type[x.second] = "PK_CLASS_uvbox";
			break;
		case PK_CLASS_uv:
			id_to_type[x.second] = "PK_CLASS_uv";
			break;
		case PK_CLASS_pointer:
			id_to_type[x.second] = "PK_CLASS_pointer";
			break;
		case PK_CLASS_vector1:
			id_to_type[x.second] = "PK_CLASS_vector1";
			break;
		case PK_CLASS_size:
			id_to_type[x.second] = "PK_CLASS_size";
			break;
		case PK_CLASS_attrib:
			id_to_type[x.second] = "PK_CLASS_attrib";
			break;
		case PK_CLASS_attdef:
			id_to_type[x.second] = "PK_CLASS_attdef";
			break;
		case PK_CLASS_group:
			id_to_type[x.second] = "PK_CLASS_group";
			break;
		case PK_CLASS_transf:
			id_to_type[x.second] = "PK_CLASS_transf";
			break;
		case PK_CLASS_ki_list:
			id_to_type[x.second] = "PK_CLASS_ki_list";
			break;
		case PK_CLASS_topol:
			id_to_type[x.second] = "PK_CLASS_topol";
			break;
		case PK_CLASS_part:
			id_to_type[x.second] = "PK_CLASS_part";
			break;
		case PK_CLASS_assembly:
			id_to_type[x.second] = "PK_CLASS_assembly";
			break;
		case PK_CLASS_body:
			id_to_type[x.second] = "PK_CLASS_body";
			break;
		case PK_CLASS_instance:
			id_to_type[x.second] = "PK_CLASS_instance";
			break;
		case PK_CLASS_region:
			id_to_type[x.second] = "PK_CLASS_region";
			break;
		case PK_CLASS_shell:
			id_to_type[x.second] = "PK_CLASS_shell";
			break;
		case PK_CLASS_face:
			id_to_type[x.second] = "PK_CLASS_face";
			break;
		case PK_CLASS_loop:
			id_to_type[x.second] = "PK_CLASS_loop";
			break;
		case PK_CLASS_edge:
			id_to_type[x.second] = "PK_CLASS_edge";
			break;
		case PK_CLASS_fin:
			id_to_type[x.second] = "PK_CLASS_fin";
			break;
		case PK_CLASS_vertex:
			id_to_type[x.second] = "PK_CLASS_vertex";
			break;
		case PK_CLASS_geom:
			id_to_type[x.second] = "PK_CLASS_geom";
			break;
		case PK_CLASS_surf:
			id_to_type[x.second] = "PK_CLASS_surf";
			break;
		case PK_CLASS_plane:
			id_to_type[x.second] = "PK_CLASS_plane";
			break;
		case PK_CLASS_cyl:
			id_to_type[x.second] = "PK_CLASS_cyl";
			break;
		case PK_CLASS_cone:
			id_to_type[x.second] = "PK_CLASS_cone";
			break;
		case PK_CLASS_sphere:
			id_to_type[x.second] = "PK_CLASS_sphere";
			break;
		case PK_CLASS_torus:
			id_to_type[x.second] = "PK_CLASS_torus";
			break;
		case PK_CLASS_bsurf:
			id_to_type[x.second] = "PK_CLASS_bsurf";
			break;
		case PK_CLASS_offset:
			id_to_type[x.second] = "PK_CLASS_offset";
			break;
		case PK_CLASS_fsurf:
			id_to_type[x.second] = "PK_CLASS_fsurf";
			break;
		case PK_CLASS_swept:
			id_to_type[x.second] = "PK_CLASS_swept";
			break;
		case PK_CLASS_spun:
			id_to_type[x.second] = "PK_CLASS_spun";
			break;
		case PK_CLASS_blendsf:
			id_to_type[x.second] = "PK_CLASS_blendsf";
			break;
		case PK_CLASS_curve:
			id_to_type[x.second] = "PK_CLASS_curve";
			break;
		case PK_CLASS_line:
			id_to_type[x.second] = "PK_CLASS_line";
			break;
		case PK_CLASS_circle:
			id_to_type[x.second] = "PK_CLASS_circle";
			break;
		case PK_CLASS_ellipse:
			id_to_type[x.second] = "PK_CLASS_ellipse";
			break;
		case PK_CLASS_bcurve:
			id_to_type[x.second] = "PK_CLASS_bcurve";
			break;
		case PK_CLASS_icurve:
			id_to_type[x.second] = "PK_CLASS_icurve";
			break;
		case PK_CLASS_fcurve:
			id_to_type[x.second] = "PK_CLASS_fcurve";
			break;
		case PK_CLASS_spcurve:
			id_to_type[x.second] = "PK_CLASS_spcurve";
			break;
		case PK_CLASS_trcurve:
			id_to_type[x.second] = "PK_CLASS_trcurve";
			break;
		case PK_CLASS_cpcurve:
			id_to_type[x.second] = "PK_CLASS_cpcurve";
			break;
		case PK_CLASS_point:
			id_to_type[x.second] = "PK_CLASS_point";
			break;
		case PK_CLASS_nabox:
			id_to_type[x.second] = "PK_CLASS_nabox";
			break;
		case PK_CLASS_item:
			id_to_type[x.second] = "PK_CLASS_item";
			break;
		case PK_CLASS_appitem:
			id_to_type[x.second] = "PK_CLASS_appitem";
			break;
		case PK_CLASS_mtopol:
			id_to_type[x.second] = "PK_CLASS_mtopol";
			break;
		case PK_CLASS_mfacet:
			id_to_type[x.second] = "PK_CLASS_mfacet";
			break;
		case PK_CLASS_mfin:
			id_to_type[x.second] = "PK_CLASS_mfin";
			break;
		case PK_CLASS_mvertex:
			id_to_type[x.second] = "PK_CLASS_mvertex";
			break;
		case PK_CLASS_mesh:
			id_to_type[x.second] = "PK_CLASS_mesh";
			break;
		case PK_CLASS_pline:
			id_to_type[x.second] = "PK_CLASS_pline";
			break;
		case PK_CLASS_lattice:
			id_to_type[x.second] = "PK_CLASS_lattice";
			break;
		case PK_CLASS_ltopol:
			id_to_type[x.second] = "PK_CLASS_ltopol";
			break;
		case PK_CLASS_lrod:
			id_to_type[x.second] = "PK_CLASS_lrod";
			break;
		case PK_CLASS_lball:
			id_to_type[x.second] = "PK_CLASS_lball";
			break;
		}
	}
    return id_to_type;
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
    return d2 <= POINT_EPSILON_2;
}

Matching make_matching(std::string part1, std::string part2, bool exact) {

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

            double tol = COINCIDENCE_TOL;
            PK_FACE_is_coincident_o_t opts;
            PK_FACE_is_coincident_o_m(opts);
            PK_FACE_coi_t coi;
            PK_VECTOR_t point;

            err = PK_FACE_is_coincident(p1_face, p2_face, tol, &opts, &coi, &point);
            assert(err == PK_ERROR_no_errors); // PK_FACE_is_coincident

            if (coi == PK_FACE_coi_yes_c || coi == PK_FACE_coi_yes_reversed_c) {
                auto p1_face_id_it = p1_topo_map.find(p1_face);
                auto p2_face_id_it = p2_topo_map.find(p2_face);
                assert(p1_face_id_it != p1_topo_map.end());
                assert(p2_face_id_it != p2_topo_map.end());
                face_matches.push_back(std::make_tuple(p1_face_id_it->second, p2_face_id_it->second));
                break;
            }

            
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

                    err = PK_TOPOL_eval_mass_props(1, bool_results.bodies, MASS_PROP_TOL, &mass_prop_opts, &intersection_amount, &mass, c_of_g, m_of_i, &periphery);
                    assert(err == PK_ERROR_no_errors); // PK_TOPOL_eval_mass_props

                    err = PK_TOPOL_eval_mass_props(1, &p1_face, MASS_PROP_TOL, &mass_prop_opts, &intersection_amount, &mass, c_of_g, m_of_i, &periphery);
                    assert(err == PK_ERROR_no_errors); // PK_TOPOL_eval_mass_props
                    err = PK_TOPOL_eval_mass_props(1, &p2_face, MASS_PROP_TOL, &mass_prop_opts, &intersection_amount, &mass, c_of_g, m_of_i, &periphery);
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

                // Check the midpoint too

                PK_VECTOR_t midpoint1, midpoint2;
                err = PK_CURVE_eval(curve1, (t_int1.value[0] + t_int1.value[1]) / 2, 0, &midpoint1);
                assert(err == PK_ERROR_no_errors); // PK_CURVE_eval

                err = PK_CURVE_eval(curve2, (t_int2.value[0] + t_int2.value[1]) / 2, 0, &midpoint2);
                assert(err == PK_ERROR_no_errors); // PK_CURVE_eval

                if (points_coincident(midpoint1.coord, midpoint2.coord)) {

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

