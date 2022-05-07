import numpy as np
import cv2

from utils.matchers import FeatureMatcher, MultipleInstanceMatcher
from shapely.geometry import Polygon


def get_bbox_edges(bbox):
    '''
    Return edges of the bounding box in this order: top, right, down, left.
    
    Parameters
    ----------
    bbox : array
        bounding box, constituted of an array of shape (n_corners, 1, 2).

    Returns
    -------
    l1, l2, l3, l4 : float
        edges of the bounding box.
    '''
    l1 = np.linalg.norm(bbox[0] - bbox[1])
    l2 = np.linalg.norm(bbox[1] - bbox[2])
    l3 = np.linalg.norm(bbox[2] - bbox[3])
    l4 = np.linalg.norm(bbox[3] - bbox[0])

    return l1, l2, l3, l4


def get_bbox_diagonals(bbox):
    '''
    Return diagonals of the bounding box in this order: max diagonal and min diagonal.
    
    Parameters
    ----------
    bbox : array
        bounding box, constituted of an array of shape (n_corners, 1, 2).

    Returns
    -------
    d1, d2 : float
        diagonals of the bounding box.
    '''

    d1 = np.linalg.norm(bbox[0] - bbox[2])
    d2 = np.linalg.norm(bbox[1] - bbox[3])

    return max(d1, d2), min(d1, d2)


def valid_bbox_shape(bbox, max_distortion=1.4):
    '''
    Perform geometric filtering of a bounding box.

    Parameters
    ----------
    bbox : array
        bounding box, constituted of an array of shape (n_corners, 1, 2).
    max_distortion : float, default 1.4 
        distortion parameter: measures the threshold for the ratio between the opposing edges and the ratio between the diagonals.
   
    Returns
    -------
    bool
        weather the shape of the bounding box is valid according to the given distortion parameters.
    '''
    p1 = Polygon(np.asarray(bbox)[:, 0, :])
    if not p1.is_valid:
        return False
    
    # edges
    l1, l2, l3, l4 = get_bbox_edges(bbox)
    # diagonals
    d1, d2 = get_bbox_diagonals(bbox)

    valid_diagonal = d1 / d2 <= max_distortion
    valid_edges1 = max(l1, l3) / min(l1, l3) <= max_distortion
    valid_edges2 = max(l2, l4) / min(l2, l4) <= max_distortion

    return valid_diagonal and valid_edges1 and valid_edges2


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


def get_bbox_overlap(bbox1, bbox2):
    '''
    Find the overlap between 2 bounding boxes.
    This is done by calculating the area of the intersection of the bounding boxes and dividing it by the area of the smallest bounding box.
    '''
    p1 = Polygon(np.asarray(bbox1)[:,0,:])
    p2 = Polygon(np.asarray(bbox2)[:,0,:])

    if not (p1.is_valid and p2.is_valid):
        return 1.

    return p1.intersection(p2).area / min(p1.area, p2.area)


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


def find_bboxes(matcher_list, model_labels=None, min_match_threshold=15, max_distortion=1.4, color_distance_threshold=5, bbox_overlap_threshold=0.8):
    '''
    Filter valid bounding boxes.

    Parameters
    ----------
    matcher_list : array or array-like
        list of ``matchers.FeatureMatcher`` of length (n_models).
    model_labels : array or array-like, optional
        list of labels of the model images, used for representing output.
    min_match_threshold : int, default 15
        Minimum number of matches to consider a bounding box as valid.
    max_distortion : int, default 4
        Maximum distortion parameter as defined in ``valid_bbox_shape`` to consider a bounding box as valid.
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
    if model_labels == None:
        model_labels = [str(i) for i in range(len(matcher_list))]
    
    im_scene = matcher_list[0].im2

    bbox_props_list = []
    for matcher, model_name in zip(matcher_list, model_labels):
        
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
            valid_shape = valid_bbox_shape(bbox, max_distortion)
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
