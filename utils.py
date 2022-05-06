from audioop import avg
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matchers import FeatureMatcher, MultipleInstanceMatcher
from shapely.geometry import Polygon

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


def crop_scene(im_scene, bbox):
    bbox_int = bbox.astype(np.int32).reshape((bbox.shape[0], 2))
    bbox_int[bbox_int < 0] = 0

    a = bbox_int[0, 0]
    b = bbox_int[3, 0]
    c = bbox_int[0, 1]
    d = bbox_int[1, 1]

    return im_scene[c:d, a:b]


def get_dominant_color_hsv(im):
    '''
    Return the dominant color in HSV space of an input RGB image.
    '''

    c = np.mean(im, axis=(0, 1)).astype(np.uint8)
    c = cv2.cvtColor(c.reshape(1, 1, 3), cv2.COLOR_RGB2HSV).reshape(3)

    h = (c[0]*2) * np.pi / 180
    s = c[1] / 255
    v = c[2] / 255

    return h, s, v


def color_distance(im1, im2):
    '''
    Calculate the color distance between 2 images.
    The color distance is computed by finding the average RGB color of each image
    and by finding the distance of the two colors in HSV space.

    Parameters
    ----------
    im1 : array
        RGB image
    im2 : array
        RGB image

    Returns
    -------
    distance : float
    '''
    h1, s1, v1 = get_dominant_color_hsv(im1)
    h2, s2, v2 = get_dominant_color_hsv(im2)

    distances = (np.sin(h1)*s1*v1 - np.sin(h2)*s2*v2)**2 + \
        (np.cos(h1)*s1*v1 - np.cos(h2)*s2*v2)**2 + (v1 - v2)**2

    return distances * 100


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


def valid_bbox(bbox, edges_ratio=4, diag_ratio=2):
    '''
    Perform geometric filtering of a bounding box.

    Parameters
    ----------
    bbox : array
        bounding box, constituted of an array of shape (n_corners, 1, 2).
    edges_ratio : float, default 4 
        edge distortion parameter: measures the threshold for the ratio between the mean of opposing edges and their standard deviation.
    diag_ratio : float, default 2
        diagonal distortion parameter: measures the threshold for the ratio between the mean of the diagonals and their standard deviation.

    Returns
    -------
    bool
        weather the shape of the bounding box is valid according to the given distortion parameters.
    '''
    # edges
    l1, l2, l3, l4 = get_bbox_edges(bbox)

    # diagonals
    d1, d2 = get_bbox_diagonals(bbox)

    vert = [l1, l3]
    hor = [l2, l4]
    edges = [l1, l2, l3, l4]
    diagonals = [d1, d2]

    # first part takes care of crossing, second part of excessive distortion
    return (np.mean(diagonals) >= np.mean(edges) and
            edges_ratio*np.std(hor) <= np.mean(hor) and
            edges_ratio*np.std(vert) <= np.mean(vert) and
            diag_ratio*np.std(diagonals) <= np.mean(diagonals))


def find_matcher_matrix(im_scene_list, im_model_list, multiple_instances=True, K=15, peaks_kw={}, homography_kw={}):
    '''
    Compute the matrix of ``matcher.FeatureMatcher`` between each scene image and model image

    Parameters
    ----------
    im_scene_list : array or array-like
        list of scene images
    im_model_list : array or array-like
        list of model images
    multiple_instances : bool, default True
        find single or multiple instances of each model in each scene
    K : int, default 15
        binning dimension in pixel of the accumulator array for the barycenter votes in the GHT.
        The minimum value is 1. Used only if ``multiple_instances`` is set to True.
    peaks_kw : dict
        keyword arguments passed to ``scipy.find_peaks`` for finding the peaks in the GHT accumulator.
        Used only if ``multiple_instances`` is set to True.
    homography_kw : dict
        keyword arguments passed to ``matcher.FeatureMatcher.set_homography_parameters``.

    Returns
    -------
    matcher_matrix : array of ``matcher.FeatureMatcher`` or ``matcher.MultipleInstacneMatcher``
        the shape is (n_scenes, n_models).
        If ``multiple_instances`` is set to False, the type of the array elements is ``matcher.FeatureMatcher``
        If ``multiple_instances`` is set to True, the type of the array elements is ``matcher.MultipleInstacneMatcher``
    '''
    # Find salient points of the images and corresponding descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    kp_scene_list, des_scene_list = sift.compute(
        im_scene_list, sift.detect(im_scene_list))
    kp_model_list, des_model_list = sift.compute(
        im_model_list, sift.detect(im_model_list))

    # The matrix is instantiated
    matcher_matrix = np.zeros(
        (len(im_scene_list), len(im_model_list)), dtype=object)

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
            # set the previously computed descriptors and keypoints for performance reasons
            matcher.set_descriptors_1(kp_model, des_model)
            matcher.set_descriptors_2(kp_scene, des_scene)
            matcher.find_matches()
            matcher_matrix[i][j] = matcher

    return matcher_matrix


def find_bboxes(matcher_list, model_filenames=None, min_match_threshold=15, max_distortion=4, color_distance_threshold=5, bbox_overlap_threshold=0.8):
    '''
    Filter valid bounding boxes.

    Parameters
    ----------
    matcher_list : array or array-like
        Array of ``matchers.FeatureMatcher`` of shape (n_models).
    model_filenames : array or array-like, optional
        Array of filenames of the model images, used for representing output.
    min_match_threshold : int, default 15
        Minimum number of matches to consider a bounding box as valid.
    max_distortion : int, default 4
        Aaximum distortion parameter as defined in ``valid_bbox`` to consider a bounding box as valid.
    color_distance_threshold : float, default 5
        Average color distance in HSV space to filter false positive bounding boxes.
    bbox_overlap_threshold : float, default 0.8
        Ratio of the area of the intersection between 2 bounding boxes and the smallest of the 2 bounding boxes.
        Used for the filtering of overlapping bounding boxes.

    Returns
    -------
    list of dict
        Each element of the list has the following attributes:
            model : string
                The name of the model image-
            corners : array
                coordinates in pixels of the corners of the bounding box. Its shape is (4, 1, 2,).
            center : array
                coordinates in pixels of center of the bounding box. Its shape is (2,).
            match_number : int
                Number of matches used to compute the bounding box.
            sufficient_matches : bool
                True if ``match_number`` is more than ``min_match_threshold``.
            valid_shape : bool
                True if the shape of the bounding box is valid according to ``max_distortion``.
            color_distance : float
                Distance of the average color distance between the model image and the scene image in the bounding box.
            valid_color : bool
                True if ``color_distance`` is smaller than ``color_distance_threshold``.
            valid_bbox : bool
                True if ``sufficient_matches``, ``valid_color``, ``valid_shape`` are true and if the bounding box does not overlap
                with another bounding box with more matches.
    '''
    if model_filenames == None:
        model_filenames = [str(i) for i in range(len(matcher_list))]
    
    im_scene = matcher_list[0].im2

    bbox_props_list = []
    for matcher, model_name in zip(matcher_list, model_filenames):
        
        im_model = matcher.im1

        # find the corners of the model
        h, w = im_model.shape[0], im_model.shape[1]
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

        # differentiate between MultipleInstanceMatcher and FeatureMatcher
        if isinstance(matcher, MultipleInstanceMatcher):
            homographies, used_kp_list = matcher.get_homographies()
        elif isinstance(matcher, FeatureMatcher):
            homographies, _ = matcher.get_homography()
            homographies = [homographies]
            used_kp_list = [len(matcher.get_matches())]
        else:
            raise TypeError(
                "Matcher must be an instance of matcher.FeatureMatcher")

        for i, (M, used_kp) in enumerate(zip(homographies, used_kp_list)):
            # Project the corners of the model onto the scene image
            bbox = cv2.perspectiveTransform(pts, M)

            high_kp = used_kp >= min_match_threshold
            valid_shape = valid_bbox(bbox, max_distortion, max_distortion/2)
            avg_color_distance = color_distance(im_model, crop_scene(im_scene, bbox))
            valid_color = avg_color_distance <= color_distance_threshold
            center = cv2.perspectiveTransform(np.float32([[[w//2, h//2]]]), M).ravel()

            bbox_props_list.append({
                'model': model_name,
                'corners': bbox,
                'center': center,
                'match_number': used_kp,
                'sufficient_matches': high_kp,
                'valid_shape': valid_shape,
                'color_distance': avg_color_distance,
                'valid_color': valid_color,
                'valid_bbox': high_kp and valid_shape and valid_color,
            })

    return filter_overlap(bbox_props_list, bbox_overlap_threshold)


def get_bbox_overlap(bbox1, bbox2):
    '''
    Find the overlap between 2 bounding boxes.
    This is done by calculating the area of the intersection of the bounding boxes and dividing it by the area of the smallest bounding box.
    '''
    p1 = Polygon(np.asarray(bbox1)[:,0,:])
    p2 = Polygon(np.asarray(bbox2)[:,0,:])
    inters = p1.intersection(p2)

    return inters.area / min(p1.area, p2.area)


def filter_overlap(bbox_props_list, bbox_overlap_threshold=0.8):
    '''
    Filter overlapping bounding boxes

    Parameters
    ----------
    bbox_props_list: list of dict

    bbox_overlap_threshold: float, default 0.8
        ratio of the area of the intersection between 2 bounding boxes and the smallest of the 2 bounding boxes.
        Used for the filtering of overlapping bounding boxes.
    
    Returns
    -------
    list of dict
    '''
    for i in range(len(bbox_props_list)):
        if not bbox_props_list[i]['valid_bbox']: continue

        bbox = bbox_props_list[i]['corners']

        for j in range(len(bbox_props_list)):
            if i==j or not bbox_props_list[j]['valid_bbox']: continue

            bbox2 = bbox_props_list[j]['corners']
            overlap = get_bbox_overlap(bbox, bbox2)

            if overlap <= bbox_overlap_threshold: continue

            if bbox_props_list[i]['match_number'] > bbox_props_list[j]['match_number'] : 
                bbox_props_list[j]['valid_bbox'] = False

    return bbox_props_list


def rgb_to_hex(rgb):
    '''Convert an rgb tuple to a hex string.
    '''
    return '#{:0>2X}{:0>2X}{:0>2X}'.format(rgb[0], rgb[1], rgb[2])


def annotate_bboxes(ax, bbox_props_list, annotation_offset, show_matches, draw_invalid_bbox):
    '''
    Put the model filenames onto the corresponding bounding box.

    Parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot
        axes on which to annotate
    bbox_props_list: list of dict
    annotation_offset: int
        offset in pixels of the label of the bounding box.
    show_matches: bool
        print match number alongside model name
    '''
    d = draw_invalid_bbox
    a = annotation_offset

    for bbox_props in bbox_props_list:
        model_name = bbox_props['model'].split('.')[0]
        match_number = bbox_props['match_number']
        center = bbox_props['center']

        if show_matches:
            ann = f"{model_name}: {match_number} m."
        else:
            ann = model_name

        # valid bboxes
        if bbox_props['valid_bbox']:
            if not 0 in d:
                continue
            ann_color = 'k'
            facecolor = rgb_to_hex(Colors.GREEN)
            alpha = 0.7
        else:
            # filtered by overlap
            if bbox_props['sufficient_matches'] and bbox_props['valid_shape'] and bbox_props['valid_color'] and 1 in d:
                ann_color = 'w'
                facecolor = rgb_to_hex(Colors.BLUE)
                alpha = 0.5

            # filtered by color
            elif bbox_props['sufficient_matches'] and bbox_props['valid_shape'] and not (bbox_props['valid_color']) and 2 in d:
                ann_color = 'k'
                facecolor = rgb_to_hex(Colors.YELLOW)
                alpha = 0.5

            #do not annotate if neither condition is satisfied
            else:
                continue

        ax.annotate(ann, center-np.array([a, 0]), color=ann_color, fontweight='bold', alpha = 0.8, fontsize=10, bbox={'facecolor': facecolor, 'alpha': alpha, 'boxstyle': 'round'})


def visualize_detections(im_scene,
                         bbox_props_list,
                         draw_invalid_bbox=0,
                         plot_height=-1,
                         annotate=True,
                         annotation_offset=30,
                         show_matches=False,
                         ax = None,
                         axes_off = False,
                         fill_bbox = True):
    '''
    Visualize the detected models with annotated bounding boxes on the scene images.

    Parameters
    ----------
    im_scene: array
        Scene image
    bbox_props_list:
        List containing the properties of the bounding boxes of the scene image
    draw_invalid_bbox: int or tuple of int, default 0
        Flag on which bounding boxes to draw. Possible values:
            0: valid bounding boxes (green, labeled)
            1: bounding boxes filtered by overlap (blue, labeled)
            2: bounding boxes filtered by color (yellow, labeled)
            3: bounding boxes filtered by geometry (orange, unlabeled)
            4: bounding boxes filtered by match number (red, unlabeled).
            5: draw all bounding boxes
    plot_height: int, optional
        Plot height in pixels. If not given, the plot will have the same size as the scene image.
    annotate: bool, default True
        Display filenames of the scene and model images.
    annotation_offset: int, default 30
        Offset in pixels of the label of the bounding box.
    show_matches: bool, default False
        Print match number alongside model name
    ax: ``matplotlib.axes._subplots.AxesSubplot``, optional
        The axes on which to show the plot
    axes_off: bool, default False
        Toggles axes ticks on plot
    fill_bbox: bool, default True

    Returns
    -------
    if ``ax`` is not provided:
    ``matplotlib.figure.Figure``, ``matplotlib.axes._subplots.AxesSubplot``

    if ``ax`` is provided:
    ``matplotlib.axes._subplots.AxesSubplot``
    '''
    d = draw_invalid_bbox
    if not type(d) is tuple:
        d = (d,)
    if 5 in d:
        d = (0,1,2,3,4)

    create_axes = ax==None

    if create_axes:
        if plot_height == -1:
            h = im_scene.shape[0]
            w = im_scene.shape[1]
        else:
            # width and height like the first scene image
            height = im_scene.shape[0]
            width = im_scene.shape[1]
            h = plot_height
            w = plot_height*width/height
        
        # instantiate subplots
        dpi = 150
        fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)

    im_scene = np.copy(im_scene)
    im1 = np.zeros_like(im_scene)  # for overlay with filled bounding boxes

    for bbox_props in bbox_props_list:

        bbox = bbox_props['corners']
        to_fill = True

        # valid bboxes
        if bbox_props['valid_bbox']:
            if not 0 in d:
                continue
            color = Colors.GREEN
        else:
            # filtered by overlap
            if bbox_props['sufficient_matches'] and bbox_props['valid_shape'] and bbox_props['valid_color'] and 1 in d:
                color = Colors.BLUE

            # filtered by color
            elif bbox_props['sufficient_matches'] and bbox_props['valid_shape'] and not (bbox_props['valid_color']) and 2 in d:
                color = Colors.YELLOW

            # filtered by shape
            elif bbox_props['sufficient_matches'] and not (bbox_props['valid_shape'] or bbox_props['valid_color']) and 3 in d:
                color = Colors.ORANGE
                to_fill = False

            # filtered by match number
            elif not bbox_props['sufficient_matches'] and 4 in d:
                color = Colors.RED
                to_fill = False

            #do not draw if neither condition is satisfied
            else:
                continue

        im_scene = cv2.polylines(im_scene, [np.int32(bbox)], True, color, 10, cv2.FILLED)
        if to_fill:
            im1 = cv2.fillPoly(im1, [np.int32(bbox)], color, cv2.LINE_8)
    
    # display scene image
    ax.imshow(im_scene)
    if fill_bbox:
        ax.imshow(im1, alpha=0.3)

    if annotate:
        annotate_bboxes(ax, bbox_props_list, annotation_offset, show_matches, d)

    if axes_off:
        ax.set_axis_off()
    
    if create_axes:
        fig.tight_layout(pad=0.5)
        return fig, ax
    
    return ax


def print_detections(bbox_props):

    model_names_find = np.unique(
        np.array([props['model'] for props in bbox_props]))

    for model_name in model_names_find:
        counter = 0
        positions = []

        for bbox_prop in bbox_props:

            if (bbox_prop['model'] != model_name or not bbox_prop['valid_bbox']):
                continue

            counter += 1

            l1, l2, l3, l4 = get_bbox_edges(bbox_prop['corners'])

            positions.append({
                'c': bbox_prop['center'],
                'w': np.mean([l2, l4]),
                'h': np.mean([l1, l3])
            }
            )

        if len(positions) > 0:
            model_name = model_name.split('.')[0]
            print(f'\tProduct {model_name} - {counter} instance found:')
            for k, pos in enumerate(positions):
                print(
                    f"\t\tInstance {k + 1}: (position: ({pos['c'][0]:.0f}, {pos['c'][1]:.0f}), width: {pos['w']:.0f}px, height: {pos['h']:.0f}px)")
                