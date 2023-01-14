import numpy as np
from matplotlib.colors import hsv_to_rgb
from PIL import Image, ImageColor

from . import rendering

plot_colors = {
    'true_positive':'#468CB8',
    'true_negative':'#61B5CF',
    'missed':'#D6CFB8',
    'incorrect':'#D4756C',
    'false_positive':'#BA5050'
}

def make_colormap(N, colors):
    c1,c2 = map(np.array, colors)
    l = np.linspace(0,1,N).reshape((-1,1))
    return hsv_to_rgb(c1*l + c2*(1-l))[:,:3]

def render_predictions(
    data, 
    face_match_preds = None, 
    edge_match_preds = None, 
    vertex_match_preds = None,
    prioritize_missing = False,
    false_match_colormap = ([0.083, 1.0, 1.0], [.167, 1.0, 1.0]),
    missing_match_colormap = ([0.75, 1.0, 0.2], [1.0, 1.0, 1.0]),
    true_match_colormap = ([0.33, 1.0, 0.7], [0.67, 1.0, 1.0]),
    unmatched_color = [0.66, 0.66, 0.66],
    renderer=None,
    render_params=rendering.RendererParams(400,400)
):
    left_V = data.left_V
    left_F = data.left_F
    left_V_to_vertices = data.left_V_to_vertices
    left_F_to_faces = data.left_F_to_faces
    left_E_to_edges = data.left_E_to_edges

    left_num_faces = data.left_faces.shape[0]
    left_num_edges = data.left_edges.shape[0]
    left_num_verts = data.left_vertices.shape[0]

    right_V = data.right_V
    right_F = data.right_F
    right_V_to_vertices = data.right_V_to_vertices
    right_F_to_faces = data.right_F_to_faces
    right_E_to_edges = data.right_E_to_edges

    face_matches = data.faces_matches
    edge_matches = data.edges_matches
    vertex_matches = data.vertices_matches

    right_num_faces = data.right_faces.shape[0]
    right_num_edges = data.right_edges.shape[0]
    right_num_verts = data.right_vertices.shape[0]

    if face_match_preds is None:
        face_match_preds = face_matches.clone()
    if edge_match_preds is None:
        edge_match_preds = edge_matches.clone()
    if vertex_match_preds is None:
        vertex_match_preds = vertex_matches.clone()

    gt_matches = {(a.item(),b.item()) for (a,b) in face_matches.T}
    pred_matches = {(a.item(),b.item()) for (a,b) in face_match_preds.T}

    true_matches = np.array(list(gt_matches.intersection(pred_matches)))
    missed_matches = np.array(list(gt_matches - pred_matches))
    false_matches = np.array(list(pred_matches - gt_matches))

    left_face_colors = np.stack([unmatched_color]*left_num_faces)
    right_face_colors = np.stack([unmatched_color]*right_num_faces)

    num_false_matches = len(false_matches)
    num_missed_matches = len(missed_matches)
    num_true_matches = len(true_matches)

    false_match_colors = make_colormap(num_false_matches, false_match_colormap)
    missed_match_colors = make_colormap(num_missed_matches, missing_match_colormap)
    true_match_colors = make_colormap(num_true_matches, true_match_colormap)

    if num_true_matches > 0:
        left_face_colors[true_matches[:,0]] = true_match_colors
        right_face_colors[true_matches[:,1]] = true_match_colors

    if num_missed_matches > 0:
        left_face_colors[missed_matches[:,0]] = missed_match_colors
        right_face_colors[missed_matches[:,1]] = missed_match_colors

    if num_false_matches > 0:
        left_face_colors[false_matches[:,0]] = false_match_colors
        right_face_colors[false_matches[:,1]] = false_match_colors
    
    if prioritize_missing and num_missed_matches > 0:
        left_face_colors[missed_matches[:,0]] = missed_match_colors
        right_face_colors[missed_matches[:,1]] = missed_match_colors

    left_F_colors = left_face_colors[left_F_to_faces.flatten()]
    right_F_colors = right_face_colors[right_F_to_faces.flatten()]

    left_poses = rendering.get_corner_poses(left_V.numpy())
    right_poses = rendering.get_corner_poses(right_V.numpy())
    left_images = [rendering.render_segmented_mesh(left_V.numpy(), left_F.T.numpy(), left_F_to_faces.flatten().numpy(), 
        id_color=left_face_colors,
        camera_params=pose, 
        norm_center=0.0, 
        norm_scale=1.0,
        renderer=renderer,
        render_params=render_params) for pose in left_poses]
    right_images = [rendering.render_segmented_mesh(right_V.numpy(), right_F.T.numpy(), right_F_to_faces.flatten().numpy(), 
        id_color=right_face_colors,
        camera_params=pose, 
        norm_center=0.0, 
        norm_scale=1.0,
        renderer=renderer,
        render_params=render_params) for pose in left_poses]
    image = rendering.grid_images(np.array([left_images,right_images]))
    return image

def show_image(im):
    return Image.fromarray(im.astype(np.uint8))