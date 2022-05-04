import numpy as np
import matplotlib.pyplot as plt
import cv2
from matchers import FeatureMatcher, MultipleInstanceMatcher

class Colors:
    # RGB values setting for colors
    RED = (255, 40, 30)
    GREEN = (50, 255, 30)
    BLUE = (10, 127, 255)

    ORANGE = (255, 127, 20)
    YELLOW = (255, 255, 0)

    BLACK = (1, 1, 1)
    WHITE = (255, 255, 255)
    GRAY = (127, 127, 127)

def get_bbox_edges(bbox):
    
    l1 = np.linalg.norm(bbox[0] - bbox[1])
    l2 = np.linalg.norm(bbox[1] - bbox[2])
    l3 = np.linalg.norm(bbox[2] - bbox[3])
    l4 = np.linalg.norm(bbox[3] - bbox[0])
    
    return l1, l2, l3, l4

def get_bbox_diagonals(bbox):
    
    d1 = np.linalg.norm(bbox[0] - bbox[2])
    d2 = np.linalg.norm(bbox[1] - bbox[3])
    
    return d1, d2

def valid_bbox(bbox, edges_ratio = 4, diag_ratio = 2):
    '''
    Function to assess if the bounding box is valid
    Parameters
    ----------
    edges_ratio: float, default 4 
        edge distortion parameter: measures the threshold for the ratio between the mean of opposing edges and their standard deviation.
    diag_ratio: float, default 2
        diagonal distortion parameter: measures the threshold for the ratio between the mean of the diagonals and their standard deviation.
    Returns
    -------
    bool
    '''
    
    # edges
    l1, l2, l3, l4 = get_bbox_edges(bbox)

    # diagonals
    d1, d2 = get_bbox_diagonals(bbox)

    vert = [l1, l3]
    hor = [l2, l4]
    edges = [l1,l2,l3,l4]
    diagonals = [d1, d2]

    # first part takes care of crossing, second part of excessive distortion
    return (np.mean(diagonals) >= np.mean(edges) and 
            edges_ratio*np.std(hor) <= np.mean(hor) and
            edges_ratio*np.std(vert) <= np.mean(vert) and 
            diag_ratio*np.std(diagonals) <= np.mean(diagonals))

def find_matcher_matrix(im_scene_list, im_model_list, multiple_instances=True, K=15, peaks_kw={}, homography_kw={}):
    '''Computes the matrix of ``matcher.FeatureMatcher`` between each scene image and model image

    Parameters
    ----------
    im_scene_list: array or array-like
        list of scene images
    im_model_list: array or array-like
        list of model images
    multiple_instances: bool, default True
        find single or multiple instances of each model in each scene
    K: int, default 15
        binning dimension in pixel of the accumulator array for the barycenter votes in the GHT.
        The minimum value is 1. Used only if ``multiple_instances`` is set to True.
    peaks_kw:
        keyword arguments passed to ``scipy.find_peaks`` for finding the peaks in the GHT accumulator.
        Used only if ``multiple_instances`` is set to True.
    homography_kw:
        keyword arguments passed to ``matcher.FeatureMatcher.set_homography_parameters``.
    
    Returns
    -------
    2D array of shape(n_scenes, n_models) of ``matcher.FeatureMatcher`` if ``multiple_instances`` is set to False
    or ``matcher.MultipleInstacneMatcher`` if ``multiple_instances`` is set to True.
    '''
    
    # Find salient points of the images and corresponding descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    kp_scene_list, des_scene_list = sift.compute(im_scene_list, sift.detect(im_scene_list))
    kp_model_list, des_model_list = sift.compute(im_model_list, sift.detect(im_model_list))

    # The matrix is instantiated
    matcher_matrix = np.zeros((len(im_scene_list), len(im_model_list)), dtype = object)

    # The matrix is populated with matches
    for i, (im_scene, kp_scene, des_scene) in enumerate(zip(im_scene_list, kp_scene_list, des_scene_list)):
        for j, (im_model, kp_model, des_model) in enumerate(zip(im_model_list, kp_model_list, des_model_list)):
            if multiple_instances:
                matcher = MultipleInstanceMatcher(im_model, im_scene)
                matcher.set_K(K)
                matcher.set_peaks_kw(**peaks_kw)
            else:
                matcher = FeatureMatcher(im_model, im_scene)
            matcher.set_homography_parameters(**homography_kw)
            #set the previously computed descriptors and keypoints for performance reasons
            matcher.set_descriptors_1(kp_model, des_model)
            matcher.set_descriptors_2(kp_scene, des_scene)
            matcher.find_matches()
            matcher_matrix[i][j] = matcher

    return matcher_matrix

def visualize_detections(matcher_matrix,
                        scene_filenames=None,
                        model_filenames=None,
                        min_match_threshold=15,
                        max_distortion=4,
                        draw_invalid_bbox=0,
                        dimension=1000,
                        vertical_layout=True,
                        annotate = True,
                        annotation_offset = 30,
                        show_matches = False):
    
    '''Visualize the detected models with annotated bounding boxes on the scene images.
    
    Parameters
    ----------
    matcher_matrix: 2D array or array-like
        array of ``matchers.FeatureMatcher`` of shape (n_scenes, n_models).
    scene_filenames: array or array-like, optional
        array of filenames of the scene images, used for visualization.
    model_filenames: array or array-like, optional
        array of filenames of the model images, used for visualization.
    min_match_threshold: int, default 15
        minimum number of matches to consider a bounding box as valid.
    max_distortion: int, default 4
        maximum distortion parameter as defined in ``valid_bbox`` to consider a bounding box as valid.
    draw_invalid_bbox: int, default 0
        possible values:
            0: draw only valid bounding boxes
            1: draw valid bboxes and distorted bboxes with at least ``min_match_threshold`` matches
            2: draw valid bboxes and undistorted bboxes with less than ``min_match_threshold`` matches
            3: draw all bounding boxes.
    dimension: int, default 1000
        plot dimension in pixels.
        If ``vertical_layout`` is True, dimension refers to the width.
        If ``vertical_layout`` is False, dimension refers to the height.
    vertical_layout: bool, default True
        vertical or horizontal stacking of scene images.
    annotate: bool, default True
        display filenames of the scene and model images.
    annotation_offset: int, default 30
        offset in pixels of the annotation of the model filename onto the homography.
    show_matches: bool, default False
        print match number alongside model name
    
    Returns
    -------
    ``matplotlib.figure.Figure``, array of ``matplotlib.axes._subplots.AxesSubplot``

    Raises
    ------
    TypeError
        if the elements in ``matcher_matrix`` are not instances of ``matcher.FeatureMatcher``
    '''
    
    n_scenes, n_models = matcher_matrix.shape

    if scene_filenames is None:
        scene_filenames = range(n_scenes)

    if model_filenames is None:
        model_filenames = range(n_models)
    
    d = draw_invalid_bbox

    # width and height like the first scene image
    height = matcher_matrix[0][0].im2.shape[0]
    width = matcher_matrix[0][0].im2.shape[1]

    if vertical_layout:
        w = dimension
        h = w*n_scenes*height/width
        subplots_kwargs = {'nrows': n_scenes}
    else:
        h = dimension
        w = h*n_scenes*width/height
        subplots_kwargs = {'ncols': n_scenes}

    # instantiate subplots
    dpi = 150
    fig, axs = plt.subplots(figsize = (w/dpi, h/dpi), dpi = dpi, **subplots_kwargs)

    if n_scenes == 1: axs = [axs]

    for i, line in enumerate(matcher_matrix):
        im_scene = np.copy(matcher_matrix[i][0].im2)
        im1 = np.zeros_like(im_scene) # for overlay with filled bounding boxes
        
        # to keep track of center positions and match number for annotations
        centers_list = []
        matches_list = []

        for j, matcher in enumerate(line):
            im_model = np.copy(matcher_matrix[i][j].im1)

            # find the corners of the model
            h, w = im_model.shape[0], im_model.shape[1]
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            
            centers = [] # centers of the bounding boxes for the current model
            matches = [] # matches used for the bounding boxes for the current model

            # differentiate between MultipleInstanceMatcher and FeatureMatcher
            if isinstance(matcher, MultipleInstanceMatcher):
                M, used_kp = matcher.get_homographies()
            elif isinstance(matcher, FeatureMatcher):
                M, _ = matcher.get_homography()
                M = [M]
                used_kp = [len(matcher.get_matches())]
            else:
                raise TypeError("Matcher must be an instance of matcher.FeatureMatcher")

            for M, used_kp in zip(M, used_kp):
                # Project the corners of the model onto the scene image
                dst = cv2.perspectiveTransform(pts, M)

                high_kp = used_kp >= min_match_threshold
                undistorted = valid_bbox(dst, max_distortion)

                # Set bounding box parameters
                # normal bounding box
                if high_kp and undistorted:
                    color = Colors.GREEN
                    width = 15
                    # fill bounding box
                    centers.append(cv2.perspectiveTransform(np.float32([[[w//2, h//2]]]), M).ravel())
                    matches.append(used_kp)
                    im1 = cv2.fillPoly(im1, [np.int32(dst)], color, cv2.LINE_8)
                    im_scene = cv2.polylines(im_scene, [np.int32(dst)], True, Colors.GREEN, 15, cv2.FILLED)

                # distorted bounding box with high number of matches
                if high_kp and not undistorted and (d==1 or d==3):
                    im_scene = cv2.polylines(im_scene, [np.int32(dst)], True, Colors.ORANGE, 10, cv2.FILLED)

                # bounding box with low number of matches
                if not high_kp and (d==2 or d==3):
                    im_scene = cv2.polylines(im_scene, [np.int32(dst)], True, Colors.RED, 10, cv2.FILLED)
                

            # save centers
            centers_list.append(centers)
            matches_list.append(matches)

        # display scene image
        axs[i].imshow(im_scene)
        axs[i].imshow(im1, alpha = 0.3)

        if annotate:
            # put annotations for model filenames
            for (model_filename, centers, matches) in zip(model_filenames, centers_list, matches_list):
                if type(model_filename) == str:
                    model_number = model_filename.split('.')[0]
                else:
                    model_number = model_filename

                for center, match_number in zip(centers, matches):
                    a = annotation_offset
                    if show_matches:
                        ann = f"{model_number}: {match_number} m."
                    else:
                        ann = model_number
                    axs[i].annotate(ann, center-np.array([a,0]), color = 'k', fontweight='bold', fontsize=10)

    for scene_filename, ax in zip(scene_filenames, axs.ravel()):
        if annotate: ax.set_title(scene_filename)
        ax.set_axis_off()

    fig.tight_layout(pad = 1.5)
    return fig, axs


def print_detections(matcher_matrix, scene_filenames=None, model_filenames=None, min_match_threshold=15, max_distortion=4):
    
    n_scenes, n_models = matcher_matrix.shape
    
    if scene_filenames is None:
        scene_filenames = range(n_scenes)

    if model_filenames is None:
        model_filenames = range(n_models)
    
    for i, line in enumerate(matcher_matrix):
        print(f"\nScene: {scene_filenames[i]}")
        
        for j, matcher in enumerate(line):
            
            im_model = matcher_matrix[i][j].im1

            # find the corners of the model
            h, w = im_model.shape[0], im_model.shape[1]
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            
            # differentiate between MultipleInstanceMatcher and FeatureMatcher
            if isinstance(matcher, MultipleInstanceMatcher):
                M, used_kp = matcher.get_homographies()
            elif isinstance(matcher, FeatureMatcher):
                M, _ = matcher.get_homography()
                M = [M]
                used_kp = [len(matcher.get_matches())]
            else:
                raise TypeError(
                    "Matcher must be an instance of matcher.FeatureMatcher")
            
            positions = []
            
            for M, used_kp in zip(M, used_kp):
                # Project the corners of the model onto the scene image
                dst = cv2.perspectiveTransform(pts, M)

                high_kp = used_kp >= min_match_threshold
                undistorted = valid_bbox(dst, max_distortion)

                # Set bounding box parameters
                # normal bounding box
                if high_kp and undistorted:
                    
                    l1, l2, l3, l4 = get_bbox_edges(dst)
                    
                    positions.append({
                        'c' : cv2.perspectiveTransform(np.float32([[[w//2, h//2]]]), M).ravel(),
                        'w' : np.mean([l2, l4]),
                        'h' : np.mean([l1, l3])
                        }
                    )
                    
            if len(positions) > 0:
                print(f'\tProduct {model_filenames[j]} - {len(positions)} instance found:')
                for k, pos in enumerate(positions):
                    print(
                        f"\t\tInstance  {k + 1} (position: ({pos['c'][0]:.0f}, {pos['c'][1]:.0f}), width: {pos['w']:.0f}px, height: {pos['h']:.0f}px)")
            
    return None
