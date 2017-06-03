import numpy as np
import matplotlib.pyplot as plt
from fundamental_matrix_estimation import *
from numpy.linalg import inv

'''
COMPUTE_EPIPOLE computes the epipole in homogenous coordinates
given matching points in two images and the fundamental matrix
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the Fundamental matrix such that (points1)^T * F * points2 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(points1, points2, F):
    # TODO: Implement this method!
	_, _, V_T = np.linalg.svd(F)
	epipole = V_T[-1]
	z = epipole[2]
	epipole /= z
	#print 'epipole:', epipole
	return epipole
    
'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    # TODO: Implement this method!
	#print 'e2 original:', e2
	t = im2.shape[0]/2
	T = np.array([[1,0,-t], [0,1,-t], [0,0,1]])
	#print 'T:', T
	x, y, _ = T.dot(e2)
	#print 'x, y, _:', x, y, _
	r1 = x / np.sqrt(x**2 + y**2)
	r2 = y / np.sqrt(x**2 + y**2)
	R = np.array([[r1,r2,0], [-r2,r1,0], [0,0,1]])
	f, _, _ = R.dot(T).dot(e2)
	G = np.array([[1,0,0], [0,1,0], [-1/f,0,1]])
	H2 = inv(T).dot(G).dot(R).dot(T)
	e3 = H2.dot(e2)
	#print 'e3:', e3
	#print 'H2:', H2
	ex = np.array([[0,-e2[2],e2[1]], [e2[2],0,-e2[0]], [-e2[1],e2[0],0]])
	M = ex.dot(F) + e2.reshape(3,1)
	#print 'new M:', M
	W = (H2.dot(M).dot(points1.T)).T
	#print 'W:', W, W[:,2].shape, len(W)
	W /= W[:,2].reshape(len(W), 1)
	#print 'new W:', W
	b = (H2.dot(points2.T)).T
	b /= b[:,2].reshape(len(b), 1)
	b = b[:,0].reshape(len(b), 1)
	#print 'b:', b, b.shape
	a = np.linalg.lstsq(W, b)[0]
	#print 'a:', a
	HA = np.array([[a[0,0], a[1,0], a[2,0]], [0,1,0], [0,0,1]])
	#print 'HA:', HA
	H1 = HA.dot(H2).dot(M)
	return H1, H2

if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    e1 = compute_epipole(points1, points2, F)
    e2 = compute_epipole(points2, points1, F.transpose())
    print "e1", e1
    print "e2", e2

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print "H1:\n", H1
    print
    print "H2:\n", H2

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()




