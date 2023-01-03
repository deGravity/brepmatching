import functools
def compose(*fs):
    return functools.reduce(lambda f, g: lambda *a, **kw: f(g(*a, **kw)), fs)

def use_bl_exact_match_labels(data):
    data.edges_matches = data.bl_exact_edges_matches
    data.faces_matches = data.bl_exact_faces_matches
    data.vertices_matches = data.bl_exact_vertices_matches
    #data.edges_matches_batch = data.left_edges_batch[data.edges_matches[0]]
    #data.faces_matches_batch = data.left_faces_batch[data.faces_matches[0]]
    #data.vertices_matches_batch = data.left_vertices_batch[data.vertices_matches[0]]
    return data