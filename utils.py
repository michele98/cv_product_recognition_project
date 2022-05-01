import numpy as np
import cv2
from scipy.signal import find_peaks

# Used to compute the matches between two images using local invariant features
class FeatureMatcher():
    
    # Attributes
    # Attributes ending with 1 refer to models while those ending with 2 refer to scenes
    _computed = False
    _kp1, _des1 = [], []
    _kp2, _des2 = [], []
    _matches = []
    _match_distance_threshold = 0.7
    _homography = np.zeros((3,3))
    _homography_mask = []
    
    # Constructor, initialize the sift, model and scene
    def __init__(self, im1, im2):
        self._sift = cv2.xfeatures2d.SIFT_create()
        self.im1 = im1
        self.im2 = im2

    # methods
    # Set methods

    def set_match_distance_threshold(self, threshold):
        self._match_distance_threshold = threshold
    
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
        d = self._match_distance_threshold
        #if the distance between the best matches is less than d times the distance from the second best matches, keep the match
        #otherwise it is probably a false match that needs to be discarded
        self._matches = [m for m,n in matches if m.distance < d*n.distance]
        
        self._find_homography()
        self._computed = True
    
    def _find_homography(self):
        src_pts = np.float32([self._kp1[m.queryIdx].pt for m in self._matches])
        dst_pts = np.float32([self._kp2[m.trainIdx].pt for m in self._matches])
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        self._homography = M
        self._homography_mask = mask.ravel().tolist()


class MultipleInstanceMatcher(FeatureMatcher):

    _peaks_kw = {'height': 0.3, 'prominence': 0.5}
    _homographies = [np.eye(3, dtype = np.float32)]

    def __init__(self, im1, im2, K = 15, sigma = 4, min_cluster_threshold = 0):
        super().__init__(im1, im2)
        self._K = K
        self._sigma = sigma
        self._peaks_kw['distance'] = 0.2*im2.shape[1]*im2.shape[0]/K**2
        self._min_cluster_threshold = min_cluster_threshold
    
    def find_matches(self, force=False):
        super().find_matches(force)
        if len(self._matches) <= 4:
            print('Model not found! There are less than 4 matches between model and scene images.')
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

    def set_sigma(self, sigma):
        self._sigma = sigma
    
    def set_peaks_kw(self, **kwargs):
        if 'distance' in kwargs.keys():
            s = self.im2.shape[0]*self.im2.shape[1]
            kwargs['distance'] = kwargs['distance']*s/self._K**2
        self._peaks_kw = kwargs

    def get_homographies(self):
        return self._homographies

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
        self._theta = kp_scene_angle - kp_model_angle

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

        k = np.ceil(3*self._sigma).astype(int)
        acc_bin = cv2.GaussianBlur(acc_bin, (2*k+1,2*k+1), self._sigma)

        self._acc_bin = acc_bin / np.max(acc_bin)

    def _find_accumulator_peaks(self):

        x = self._acc_bin.flatten()

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

        #print(np.unique(self._predicted_labels))
        for label in np.unique(self._predicted_labels):
            label_mask = self._predicted_labels==label
            votes = self._vote_pos[label_mask]
            if label==-1 or len(votes) < self._min_cluster_threshold: continue

            kp_model_filtered = self._kp_model_pos[label_mask]
            kp_scene_filtered = self._kp_scene_pos[label_mask]

            # impossible to compute homography with less than 4 points
            if len(kp_model_filtered) < 4: continue

            M, mask = cv2.findHomography(kp_model_filtered, kp_scene_filtered, cv2.RANSAC, 1.)
            if M is None: continue
            homographies.append(M)
        #print(f'Found {len(homographies)} homographies')
        self._homographies = homographies