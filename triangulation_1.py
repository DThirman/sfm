import sys
import numpy as np
import os
import scipy.misc
from scipy.optimize import least_squares
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sfm_utils import *
from numpy.linalg import inv
from glumpy.api.matplotlib import *
import cv2


'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    # TODO: Implement this method!
	#print 'E:\n', E
	U, D, V_T = np.linalg.svd(E)
	W = np.array([[0,-1,0], [1,0,0], [0,0,1]])
	Q1 = U.dot(W).dot(V_T)
	Q2 = U.dot(W.T).dot(V_T)
	R1 = np.linalg.det(Q1) * Q1
	R2 = np.linalg.det(Q2) * Q2
	#print 'R1:', R1, R2
	T1 = U[:,-1].reshape(U.shape[0], 1)
	T2 = -T1
	RT1 = np.concatenate([R1, T1], 1)
	RT2 = np.concatenate([R1, T2], 1)
	RT3 = np.concatenate([R2, T1], 1)
	RT4 = np.concatenate([R2, T2], 1)
	return RT1, RT2, RT3, RT4





'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
	P = image_points
	M = camera_matrices
	for i in range(P.shape[0]):
		ai = np.array([P[i,0]*M[i,2] - M[i,0], P[i,1]*M[i,2] - M[i,1]])
		if i == 0: A = ai
		else: A = np.concatenate([A, ai], 0)	
	#print 'A:', A, A.shape
	_, _, V_T = np.linalg.svd(A)
	P3D = V_T[-1]
	P3D /= P3D[-1]
	P3D = P3D[:3]
	#print 'P3D:', P3D
	return P3D



'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2Mx1 reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    # TODO: Implement this method!
	P3d = point_3d
	P = image_points
	M = camera_matrices
	P3d = np.concatenate([P3d, [1]], 0).reshape(4,1)
	#print 'P3d:', P3d, P3d.shape
	for i in range(P.shape[0]):
		pi = M[i].dot(P3d)
		pi = (pi/pi[-1])[:2].T
		if i == 0: P_prime = pi
		else: P_prime = np.concatenate([P_prime, pi], 0)
	#print 'p prime:', P_prime, P_prime.shape
	error = P_prime - P
	#print 'error:', error
	error = error.reshape(-1)
	#print 'error:', error
	return error

	


'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    # TODO: Implement this method!
	P3d = point_3d
	M = camera_matrices
	#print 'shapes:', P3d.shape, M.shape
	P3d = np.concatenate([P3d, [1]], 0).reshape(4,1)
	for i in range(M.shape[0]):
		pi = M[i].dot(P3d)
		jix = ((pi[2]*M[i,0,:3] - pi[0]*M[i,2,:3]) / pi[2]**2).reshape(1,3)
		jiy = ((pi[2]*M[i,1,:3] - pi[1]*M[i,2,:3]) / pi[2]**2).reshape(1,3)
		if i == 0: J = np.concatenate([jix, jiy], 0)
		else: J = np.concatenate([J, jix, jiy], 0)
	#print 'Jacobian:\n', J
	return J




'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
	#print 'shapes:', image_points.shape, camera_matrices.shape
	P3d = linear_estimate_3d_point(image_points, camera_matrices)
	for i in range(10):
		J = jacobian(P3d, camera_matrices)
		e = reprojection_error(P3d, image_points, camera_matrices)
		P3d = P3d - inv(J.T.dot(J)).dot(J.T).dot(e)
	#print 'final P3d:', P3d
	return P3d




'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    # TODO: Implement this method!
	P = image_points
	RTs = estimate_initial_RT(E)
	zeros = np.zeros((3,1))
	M1 = np.concatenate([K, zeros], 1)
	max_count = 0
	for RT in RTs:
		positive_count = 0
		M2 = K.dot(RT)
		for pt in P:
			M = np.array([M1, M2])
			P3d1 = nonlinear_estimate_3d_point(pt, M)
			P3d1h = np.concatenate([P3d1, [1]], 0).reshape(4,1)
			P3d2 = RT.dot(P3d1h)
			if P3d1[2] >= 0 and P3d2[2] >= 0:
				positive_count += 1
				if positive_count > max_count:
					max_count = positive_count
					RT_true = RT
		#print 'positive count for this RT:', positive_count
	#print 'true RT has this many positive points:', max_count
	return RT_true
	

def display(points):
	# Create a new figure
	figure = Figure((24,24))

	# Create a subplot on left, using trackball interface (3d)
	size = 20
	left = figure.add_axes( [0.010, 0.01, 0.98, 0.98],
				xscale = LinearScale(domain=[-size,size], range=[-size,size]),
				yscale = LinearScale(domain=[-size,size], range=[-size,size]),
				zscale = LinearScale(domain=[-size,size], range=[-size,size]),
				interface = Trackball(name="trackball"),
				facecolor=(0,1,0,0.25), aspect=1 )

	# Create a new collection of points
	collection = PointCollection("agg")

	# Add a view of the collection on the left subplot
	left.add_drawable(collection)

	# Add some points
	#points = np.random.uniform(-1,1,(10000,3))
	collection.append(points)
	#print 'points:', points.shape

	# Show figure
	figure.show()



'''
REFINE_MATCH: Filter out spurious matches between two images by using RANSAC
to find a projection matrix. 

Arguments:
    keypoints1 - Keypoints in the first image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_1, 4).

    keypoints2 - Keypoints in the second image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_2, 4).

    matches - An int ndarray of size (N, 2) of indices that indicate what
        keypoints from the first image (keypoints1)  match with the second 
        image (keypoints2). For example, [7 5] would mean that the keypoint at
        index 7 of keypoints1 matches the keypoint at index 5 of keypoints2). 
        Not every keypoint will necessarily have a  match, so N is not the same
        as the number of rows in keypoints1 or keypoints2. 

    reprojection_threshold - If the reprojection error is below this threshold,
        then we will count it as an inlier during the RANSAC process.

    num_iterations - The number of iterations we will run RANSAC for.

Returns:
    inliers - A vector of integer indices that correspond to the inliers of the
        final model found by RANSAC.

    model - The projection matrix H found by RANSAC that has the most number of
        inliers.
'''
def ransac(keypoints1, keypoints2, matches, reprojection_threshold = 150,
        num_iterations = 100):
    # TODO: Implement this method!
	#print 'keypoints:', keypoints1.shape, keypoints2.shape, matches.shape
	max_inliers_count = 0
	for i in range(num_iterations):
		if i % 100 == 0:
			print ' iteration:', i
		#print 'iteration:', i
		indices = np.random.randint(0, len(matches), 4)
		first = True
		for i in indices:
			i1 = matches[i].queryIdx
			i2 = matches[i].trainIdx
			p1 = keypoints1[i1].pt
			p2 = keypoints2[i2].pt
			p1h = np.concatenate([p1, [1]], 0).reshape(1,3)
			p2h = np.concatenate([p2, [1]], 0).reshape(1,3)
			if first:
				x1 = p1h
				x2 = p2h
				first = False
			else:
				x1 = np.concatenate([x1, p1h], 0)
				x2 = np.concatenate([x2, p2h], 0)
		#print x1, x2, x1.shape, x2.shape
		H = np.linalg.lstsq(x1, x2)[0].T
		#print 'H:\n', H, H.shape
		inliers_count = 0
		inliers = []
		for index, m in enumerate(matches):
			p1 = keypoints1[m.queryIdx].pt
			#p2 = keypoints2[m.queryIdx].pt.reshape(2,1)
			p2 = keypoints2[m.trainIdx].pt
			p1h = np.concatenate([p1, [1]], 0).reshape(3,1)
			p2h_prime = H.dot(p1h)
			p2_prime = (p2h_prime / p2h_prime[2])[:2]
			error = np.sqrt(np.sum((p2 - p2_prime)**2))
			if error <= reprojection_threshold:
				inliers_count += 1
				inliers.append(index)
		#print 'inliers:', inliers_count
		if inliers_count > max_inliers_count:
			max_inliers_count = inliers_count
			model = H
			max_inliers = inliers
	#print 'max inliers:', max_inliers, max_inliers_count
	return max_inliers, model





def sift(image_data_dir, image_paths):
	#image_data_dir = 'data/statue/'
	#image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        #sorted(os.listdir('data/statue/images')) if '.jpg' in x]
	print 'image paths 0:', image_paths[0]

	ret = []

	for i in range(len(image_paths) - 1):
	#for i in range(1):

			img1 = cv2.imread(image_paths[i])
			img2 = cv2.imread(image_paths[i + 1])

			gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
			gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

			dense = cv2.FeatureDetector_create("Dense")
			sift = cv2.SIFT()
			surf = cv2.SURF()

			'''kp1 = dense.detect(gray1)
			kp2 = dense.detect(gray2)
			kp1, des1 = sift.compute(gray1, kp1)
			kp2, des2 = sift.compute(gray2, kp2)'''

			kp1, des1 = sift.detectAndCompute(gray1, None)
			kp2, des2 = sift.detectAndCompute(gray2, None)


			print 'kp, des:', len(kp1), des1.shape, len(kp2), des2.shape
			#print 'content:', kp[0].pt, kp[0].size, kp[1].pt, kp[2].pt, dtcher()

			bf = cv2.BFMatcher()	
			matches = bf.knnMatch(des1, des2, k=2)
			m, n = matches[0]
			print 'matches:', len(matches), matches[0], m.queryIdx, m.trainIdx, m.distance, n.queryIdx, n.trainIdx, n.distance
			
			good = []
			for m, n in matches:
				if m.distance < 0.75 * n.distance:
					good.append(m)
			print '# matches before ransac;', len(good)
			
			good = np.array(good)
			
			good_idx, _ = ransac(kp1, kp2, good)
			print '# matches after ransac:', len(good_idx)

			better = good[good_idx]

			#img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)
			img3 = drawMatches(gray1,kp1,gray2,kp2, good)
			plt.imshow(img3)
			plt.show()
			img4 = drawMatches(gray1,kp1,gray2,kp2, better)
			plt.imshow(img4)
			plt.show()

			reti = np.zeros((4, len(better)))
			for i in range(len(better)):
				x1, y1 = kp1[better[i].queryIdx].pt
				x2, y2 = kp2[better[i].trainIdx].pt

				#if i == 10: print 'x1, y1:', x1, y1
				reti[:, i] = [x1, y1, x2, y2]
			ret.append(reti)


	return ret


def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0
 
    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.
 
    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.
 
    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.
 
    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """
 
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
 
    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
 
    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])
 
    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])
 
    # For each pair of points we have between both images
    # draw circles, then connect a line between them
 
    for mat in matches:
        #mat = mat[0]
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
 
        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
 
        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)  
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
 
        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
 
 
    # Show the image
 
    # Also return the image if you'd like a copy
    return out



'''def draw_matches(img1, kp1, img2, kp2, matches, color=None): 
    """Draws lines between matching keypoints of two images.  
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles 
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same 
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to 
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.  
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.  
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 15
    thickness = 2
    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color: 
            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m.trainIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.queryIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)
    
    plt.figure(figsize=(10,10))
    plt.imshow(new_img)
    plt.show()'''



if __name__ == '__main__':
	
    run_pipeline = True
    image_data_dir = 'data/hoover/'
    image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        sorted(os.listdir('data/hoover/images')) if '.jpg' in x]
    m1 = np.load(os.path.join(image_data_dir, '1.npy'))[:4,:]
    m2 = np.load(os.path.join(image_data_dir, '2.npy'))[:4,:]
    m3 = np.load(os.path.join(image_data_dir, '3.npy'))[:4,:]
    #m4 = np.load(os.path.join(image_data_dir, '4.npy'))[:4,:]
    dense_matches = [m1, m2, m3]
    print 'm1 to m4:', m1.shape, m2.shape, m3.shape
    print 'dense matches', len(dense_matches)

    #dense_matches = sift(image_data_dir, image_paths)

    #print 'dense matches', dense_matches, dense_matches.shape
    #matches_subset = dense_matches

    # Load the data
    #unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    #unit_test_image_matches = np.load('data/unit_test_image_matches.npy')

    focal_length = 719.5459
    #matches_subset = np.load(os.path.join(image_data_dir,
    #    'matches_subset.npy'))[0,:]
    matches_subset = [m1[:,:100], m2[:,:100], m3[:,:100]]
    print 'matches subset', len(matches_subset), matches_subset[0].shape

    #dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'))
    #print 'dense matches:', dense_matches[0], dense_matches.shape, dense_matches[0].shape
    fundamental_matrices = np.load(os.path.join(image_data_dir,
        'fundamental_matrices.npy'))[0,:]

    # Part A: Computing the 4 initial R,T transformations from Essential Matrix
    '''print '-' * 80
    print "Part A: Check your matrices against the example R,T"
    print '-' * 80
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    E = K.T.dot(fundamental_matrices[0]).dot(K)'''

    im0 = scipy.misc.imread(image_paths[0])
    im_height, im_width, _ = im0.shape

    '''example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
        [0.1019, 0.9948, 0.0045, -0.0089],
        [0.2041, -0.0254, 0.9786, 0.0331]])
    print "Example RT:\n", example_RT
    estimated_RT = estimate_initial_RT(E)
    print
    print "Estimated RT:\n", estimated_RT

    # Part B: Determining the best linear estimate of a 3D point
    print '-' * 80
    print 'Part B: Check that the difference from expected point '
    print 'is near zero'
    print '-' * 80
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        camera_matrices.copy())
    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    print "Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum()

    # Part C: Calculating the reprojection error and its Jacobian
    print '-' * 80
    print 'Part C: Check that the difference from expected error/Jacobian '
    print 'is near zero'
    print '-' * 80
    estimated_error = reprojection_error(
            expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    print "Error Difference: ", np.fabs(estimated_error - expected_error).sum()
    expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
         [0., 154.33943931, 36.51165089],
         [141.87950588, -14.27738422, -56.20341644],
         [21.9792766, 149.50628901, 32.23425643]])
    print "Jacobian Difference: ", np.fabs(estimated_jacobian
        - expected_jacobian).sum()

    # Part D: Determining the best nonlinear estimate of a 3D point
    print '-' * 80
    print 'Part D: Check that the reprojection error from nonlinear method'
    print 'is lower than linear method'
    print '-' * 80
    estimated_3d_point_linear = linear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    error_linear = reprojection_error(
        estimated_3d_point_linear, unit_test_image_matches,
        unit_test_camera_matrix)
    print "Linear method error:", np.linalg.norm(error_linear)
    error_nonlinear = reprojection_error(
        estimated_3d_point_nonlinear, unit_test_image_matches,
        unit_test_camera_matrix)
    print "Nonlinear method error:", np.linalg.norm(error_nonlinear)

    # Part E: Determining the correct R, T from Essential Matrix
    print '-' * 80
    print "Part E: Check your matrix against the example R,T"
    print '-' * 80
    estimated_RT = estimate_RT_from_E(E,
        np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print "Example RT:\n", example_RT
    print
    print "Estimated RT:\n", estimated_RT'''

    # Part F: Run the entire Structure from Motion pipeline
    if not run_pipeline:
        sys.exit()
    print '-' * 80
    print 'Part F: Run the entire SFM pipeline'
    print '-' * 80
    frames = [0] * (len(image_paths) - 1)
    for i in xrange(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length,
                fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])
    merged_frame = merge_all_frames(frames)

    # Construct the dense matching
    camera_matrices = np.zeros((2,3,4))
    dense_structure = np.zeros((0,3))
    for i in xrange(len(frames)-1):
    #for i in range(1):
        matches = dense_matches[i]
        camera_matrices[0,:,:] = merged_frame.K.dot(
            merged_frame.motion[i,:,:])
        camera_matrices[1,:,:] = merged_frame.K.dot(
                merged_frame.motion[i+1,:,:])
        points_3d = np.zeros((matches.shape[1], 3))
        use_point = np.array([True]*matches.shape[1])
        for j in xrange(matches.shape[1]):
            points_3d[j,:] = nonlinear_estimate_3d_point(
                matches[:,j].reshape((2,2)), camera_matrices)
        dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))
    print 'dense structure:', dense_structure.shape, np.max(dense_structure, 0), np.min(dense_structure, 0)
    
    display(dense_structure)    
 
    '''fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax.scatter(dense_structure[:,0], dense_structure[:,1], dense_structure[:,2],
        c='k', depthshade=True, s=2)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 10)
    ax.view_init(-100, 90)

    plt.show()'''

     
