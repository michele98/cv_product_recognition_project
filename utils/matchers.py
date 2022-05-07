import numpy as np
import cv2
from scipy.signal import find_peaks


class FeatureMatcher():
    '''
    Used to compute the matches between two images using local invariant features.
    '''
    # Attributes
    # Attributes ending with 1 refer to models while those ending with 2 refer to scenes
    _computed = False
    _kp1, _des1 = [], []
    _kp2, _des2 = [], []
    _matches = []
    _homography_parameters = {'match_distance_threshold': 0.7, 'ransacReprojThreshold': 1.}
    _homography = np.eye(3)
    _homography_mask = []
    
    # Constructor, initialize the sift, model and scene
    def __init__(self, im1, im2):
        self._sift = cv2.xfeatures2d.SIFT_create()
        self.im1 = im1
        self.im2 = im2

    # methods
    # Set methods

    def set_match_distance_threshold(self, threshold):
        self._homography_parameters['match_distance_threshold'] = threshold
    
    def set_ransac_reproj_threshold(self, threshold):
        self._homography_parameters['ransac_reproj_threshold'] = threshold
    
    def set_homography_parameters(self, **kwargs):
        for key in kwargs.keys():
            if key not in self._homography_parameters.keys():
                raise KeyError(f'Unknown parameter {key}. The possible homography parameters are: match_distance_threshold and ransacReprojThreshold.')

        for key, value in kwargs.items():
            self._homography_parameters[key] = value

    # Sets keypoints and descriptors for image 1
    def set_descriptors_1(self, kp, des):
        self._kp1 = kp
        self._des1 = des

    # Sets keypoints and descriptors for image 2
    def set_descriptors_2(self, kp, des):
        self._kp2 = kp
        self._des2 = des
    
    def _reset_matches(self):
        '''
        Resets the internally stored data.
        '''
        self._matches = []
        self._computed = False

    # Get methods
    # The homography is computed from the matches using the RANSAC method.
    # Note: to get the homography from image 2 and image 1, invert the transformation matrix.
    def get_homography(self):
        # ndarray, 3x3 homography matrix between image 1 and image 2
        return self._homography, self._homography_mask
    
    def get_matches(self):
        # Matches: list of cv2.DMatch, the matches between image 1 and image 2
        return self._matches
    
    def get_keypoints(self):
        # list of cv2.KeyPoint, keypoints of image 1 and 2 respectively
        return self._kp1, self._kp2

    def get_descriptors(self):
        return self._des1, self._des2
    
    # Other methods
    def _find_descriptors_1(self):
        self._kp1, self._des1 = self._sift.compute(self.im1, self._sift.detect(self.im1))

    def _find_descriptors_2(self):
        self._kp2, self._des2 = self._sift.compute(self.im2, self._sift.detect(self.im2))

    # Computes matches between model and scene
    # force: recomputes the matches if the method "find_matches" has already been called. Default is False.
    def find_matches(self, force = False):
        
        if self._computed and not force:
            print('Matches already computed. To compute them again set force=True')
            return
        
        self._reset_matches()
        
        #The salient points for the two images and the scene images are stored internally.
        #If the salient points and their descriptors are not passed using ``set_descriptors_1`` and ``set_descriptors_2``, they are computed.
        if len(self._kp1) == 0:
            self._find_descriptors_1()

        if len(self._kp2) == 0:
            self._find_descriptors_2()
        
        # Defining index for approximate kdtree algorithm
        FLANN_INDEX_KDTREE = 1

        # Defining parameters for algorithm 
        flann = cv2.FlannBasedMatcher(
            {'algorithm': FLANN_INDEX_KDTREE,
             'trees': 5},
            {'checks': 50})
        
        #compute the best 2 matches for each salient point in the query image
        matches = flann.knnMatch(self._des1, self._des2, k=2)
        d = self._homography_parameters['match_distance_threshold']
        #if the distance between the best matches is less than d times the distance from the second best matches, keep the match
        #otherwise it is probably a false match that needs to be discarded
        self._matches = [m for m,n in matches if m.distance < d*n.distance]
        
        self._find_homography()
        self._computed = True
    
    def _find_homography(self):
        src_pts = np.float32([self._kp1[m.queryIdx].pt for m in self._matches])
        dst_pts = np.float32([self._kp2[m.trainIdx].pt for m in self._matches])

        reproj_th = self._homography_parameters['ransacReprojThreshold']

        if len(src_pts)>=4:
            homography, homography_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_th)
            if not homography is None:
                self._homography = homography
                self._homography_mask = homography_mask


class MultipleInstanceMatcher(FeatureMatcher):
    '''
    Used to compute the matches between two images using local invariant features and Generalized Hough Transform.
    '''
    _peaks_kw = {'height': 0.3}
    _homographies = [np.eye(3, dtype = np.float32)]
    _used_kp = [0]

    def __init__(self, im1, im2, K = 15, min_cluster_threshold = 0):
        super().__init__(im1, im2)
        self._K = K
        self._peaks_kw['distance'] = 1
        self._min_cluster_threshold = min_cluster_threshold
    
    def find_matches(self, force=False):
        super().find_matches(force)
        if len(self._matches) <= 4:
            #print('Model not found! There are less than 4 matches between model and scene images.')
            return
        self._calculate_r_vectors()
        self._cast_votes()
        self._compute_accumulator()
        self._find_accumulator_peaks()
        self._assign_keypoints_label()
        self._find_homographies()

    # set methods
    def set_K(self, K):
        self._K = K
    
    def set_peaks_kw(self, **kwargs):
        if 'distance' in kwargs.keys():
            s = self.im2.shape[0]*self.im2.shape[1]
            kwargs['distance'] = kwargs['distance']*s/self._K**2
        self._peaks_kw = kwargs

    def get_homographies(self):
        return self._homographies, self._used_kp

    # other methods

    # set r vectors, theta, s (scale), position of key points of the scene and of the model
    def _calculate_r_vectors(self):

        #filter keypoints that match
        kp_model = [self._kp1[i] for i in [m.queryIdx for m in self._matches]]
        kp_scene = [self._kp2[i] for i in [m.trainIdx for m in self._matches]]

        #calculate r vectors for the model image
        self._kp_model_pos = np.asarray([k.pt for k in kp_model])
        self._kp_scene_pos = np.asarray([k.pt for k in kp_scene])

        kp_model_angle = np.asarray([k.angle for k in kp_model])
        kp_scene_angle = np.asarray([k.angle for k in kp_scene])
        self._theta = (kp_scene_angle - kp_model_angle) * np.pi / 180

        kp_model_size = np.asarray([k.size for k in kp_model])
        kp_scene_size = np.asarray([k.size for k in kp_scene])
        self._s = kp_scene_size/kp_model_size

        barycenter_model = np.mean(self._kp_model_pos, axis = 0)

        self._r_vectors = barycenter_model - self._kp_model_pos

    def _cast_votes(self):
        #convert the relative rotation of the keypoints in array of rotation matrices
        rotmats = np.array([[np.cos(self._theta), -np.sin(self._theta)],
                        [np.sin(self._theta), np.cos(self._theta)]])
        rotmats = np.moveaxis(rotmats, (2,0,1), (0,1,2))

        #rotate the r vectors according to their corresponding rotation matrix
        r_vectors_rotated = rotmats@np.expand_dims(self._r_vectors, axis = -1)
        r_vectors_rotated = r_vectors_rotated.squeeze()

        #scale the rotated vectors according to their corresponding scale
        r_vectors_scaled = np.multiply(r_vectors_rotated, np.expand_dims(self._s, axis = -1))

        #vote the positions of the barycenter and 
        vote_pos = self._kp_scene_pos + r_vectors_scaled
        vote_pos = np.rint(vote_pos).astype(int)

        #discard votes outside of the image frame
        vote_inside_mask = (vote_pos[:,0] < self.im2.shape[1]) & (vote_pos[:,1] < self.im2.shape[0]) & (vote_pos[:,0] >= 0) & (vote_pos[:,1] >= 0)
        self._vote_pos = vote_pos[vote_inside_mask]
        self._kp_model_pos = self._kp_model_pos[vote_inside_mask]
        self._kp_scene_pos = self._kp_scene_pos[vote_inside_mask]

    def _compute_accumulator(self):
        n_bins = self.im2.shape[1]//self._K
        m_bins = self.im2.shape[0]//self._K
        acc_bin = np.zeros((n_bins, m_bins), dtype = np.float32)

        for pos in self._vote_pos:
            i,j = pos//self._K

            #discard votes outside of the image frame
            if i>=n_bins or j>=m_bins or i<0 or j<0: continue

            acc_bin[i,j]+=1

        self._acc_bin = acc_bin / np.max(acc_bin)

    def _find_accumulator_peaks(self):

        x = self._acc_bin.flatten()
        if self._peaks_kw['distance'] == 0:
            self._peaks_kw['distance'] = 1
        peaks_1d, _ = find_peaks(x, **self._peaks_kw)

        peaks = np.vstack(np.unravel_index(peaks_1d, self._acc_bin.shape))*self._K
        self._peaks = peaks.T

    def _assign_keypoints_label(self):
        n_votes = self._vote_pos.shape[0]
        n_peaks = self._peaks.shape[0]

        # initialize array of predicted labels
        predicted_labels = np.zeros(n_votes)

        # return if there are no found peaks or just 1,
        # as all matching points should belong to the same instance
        if n_peaks == 0 or n_peaks == 1:
            self._predicted_labels = predicted_labels
            return

        # initialize and compute matrix of distances between peaks and votes.
        # First axis for the peak, second axis for the keypoint
        distances = np.zeros((n_peaks, n_votes, ))

        for i, peak in enumerate(self._peaks):
            distances[i] = np.linalg.norm(self._vote_pos - peak.T, axis = 1)

        self._predicted_labels = np.argmin(distances, axis = 0)

    def _find_homographies(self):

        homographies = []
        used_kp = []
        reproj_th = self._homography_parameters['ransacReprojThreshold']

        #print(np.unique(self._predicted_labels))
        for label in np.unique(self._predicted_labels):
            label_mask = self._predicted_labels==label
            votes = self._vote_pos[label_mask]
            if label==-1 or len(votes) < self._min_cluster_threshold: continue

            kp_model_filtered = self._kp_model_pos[label_mask]
            kp_scene_filtered = self._kp_scene_pos[label_mask]

            # impossible to compute homography with less than 4 points
            if len(kp_model_filtered) < 4: continue

            M, mask = cv2.findHomography(kp_model_filtered, kp_scene_filtered, cv2.RANSAC, reproj_th)
            if M is None: continue
            homographies.append(M)
            used_kp.append(len(kp_model_filtered[mask]))
        #print(f'Found {len(homographies)} homographies')
        self._homographies = homographies
        self._used_kp = used_kp


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
                if len(peaks_kw)!=0:
                    matcher.set_peaks_kw(**peaks_kw)
            else:
                matcher = FeatureMatcher(im_model, im_scene)
            if len(homography_kw)!=0:
                matcher.set_homography_parameters(**homography_kw)
            # set the previously computed descriptors and keypoints for performance reasons
            matcher.set_descriptors_1(kp_model, des_model)
            matcher.set_descriptors_2(kp_scene, des_scene)
            matcher.find_matches()
            matcher_matrix[i][j] = matcher

    return matcher_matrix
