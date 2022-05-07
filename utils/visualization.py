import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils.bbox_filtering import get_bbox_edges


class Colors:
    '''
    RGB values for various colors.
    '''
    RED = (255, 40, 30)
    GREEN = (50, 255, 30)
    BLUE = (10, 127, 255)

    ORANGE = (255, 127, 20)
    YELLOW = (255, 255, 0)

    BLACK = (1, 1, 1)
    WHITE = (255, 255, 255)
    GRAY = (127, 127, 127)


def rgb_to_hex(rgb):
    '''
    Convert an rgb tuple to a hex string.
    '''
    return '#{:0>2X}{:0>2X}{:0>2X}'.format(rgb[0], rgb[1], rgb[2])


def annotate_bboxes(ax, bbox_props_list, annotation_offset, show_matches, draw_invalid_bbox):
    '''
    Put the model labels onto the corresponding bounding box.

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
    im_scene : array
        Scene image
    bbox_props_list : list of dict
        List containing the properties of the bounding boxes of the scene image
    draw_invalid_bbox : int or tuple of int, default 0
        Flag on which bounding boxes to draw. Possible values:
            0: valid bounding boxes (green, labeled)
            1: bounding boxes filtered by overlap (blue, labeled)
            2: bounding boxes filtered by color (yellow, labeled)
            3: bounding boxes filtered by geometry (orange, unlabeled)
            4: bounding boxes filtered by match number (red, unlabeled).
            5: draw all bounding boxes
    plot_height : int, optional
        Plot height in pixels. If not given, the plot will have the same size as the scene image.
    annotate : bool, default True
        Display labels of the scene and model images.
    annotation_offset : int, default 30
        Offset in pixels of the label of the bounding box.
    show_matches : bool, default False
        Print match number alongside model name
    ax : ``matplotlib.axes._subplots.AxesSubplot``, optional
        The axes on which to show the plot
    axes_off : bool, default False
        Toggles axes ticks on plot
    fill_bbox : bool, default True

    Returns
    -------
    if ``ax`` is not provided :
    ``matplotlib.figure.Figure``, ``matplotlib.axes._subplots.AxesSubplot``

    if ``ax`` is provided :
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

        im_scene = cv2.polylines(im_scene, [np.int32(bbox)], True, color, 10, cv2.LINE_AA)
        if to_fill:
            im1 = cv2.fillPoly(im1, [np.int32(bbox)], color, cv2.LINE_AA)
    
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


def print_detections(bbox_props_list):
    '''
    Print detections of each model.

    Parameters
    ----------
        bbox_props_list : list of dict
        List containing the properties of the bounding boxes of the scene image
    '''
    model_names_find = np.unique(
        np.array([props['model'] for props in bbox_props_list]))

    for model_name in model_names_find:
        counter = 0
        positions = []

        for bbox_props in bbox_props_list:

            if (bbox_props['model'] != model_name or not bbox_props['valid_bbox']):
                continue

            counter += 1

            l1, l2, l3, l4 = get_bbox_edges(bbox_props['corners'])

            positions.append({
                'c': bbox_props['center'],
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
