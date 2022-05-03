from msilib.schema import Feature
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


def valid_bbox(bbox, d = 0.25):
    '''
    Function to assess if the bounding box is valid
    @param d: distortion parameter, default is 4.
            Measures the threshold for the ratio between the mean of opposing edges and their standard deviation.
    '''
    # edges
    l1 = np.linalg.norm(bbox[0] - bbox[1])
    l2 = np.linalg.norm(bbox[1] - bbox[2])
    l3 = np.linalg.norm(bbox[2] - bbox[3])
    l4 = np.linalg.norm(bbox[3] - bbox[0])

    # diagonals
    d1 = np.linalg.norm(bbox[0] - bbox[2])
    d2 = np.linalg.norm(bbox[1] - bbox[3])

    vert = [l1, l3]
    hor = [l2, l4]
    edges = [l1,l2,l3,l4]
    diagonals = [d1, d2]

    # first part takes care of crossing, second part of excessive distortion
    return np.mean(diagonals)>=np.mean(edges) and d*np.std(hor)<=np.mean(hor) and d*np.std(vert)<=np.mean(vert)


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
    ``matplotlib.figure.Figure``, ``matplotlib.axes._subplots.AxesSubplot``

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
                print(M)
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
                
                # draw bounding box on scene image

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