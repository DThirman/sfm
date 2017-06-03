import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import scipy.io as sio
from epipolar_utils import *

'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using 
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the linear least squares eight
point algorithm works
'''
def lls_eight_point_alg(points1, points2):
    # TODO: Implement this method!
	w0 = (points1[:,0] * points2[:,0]).reshape(points2.shape[0], 1)
	w1 = (points1[:,1] * points2[:,0]).reshape(points2.shape[0], 1)
	w2 = (points2[:,0]).reshape(points2.shape[0], 1)
	w3 = (points1[:,0] * points2[:,1]).reshape(points2.shape[0], 1)
	w4 = (points1[:,1] * points2[:,1]).reshape(points2.shape[0], 1)
	w5 = (points2[:,1]).reshape(points2.shape[0], 1)
	w6 = (points1[:,0]).reshape(points2.shape[0], 1)
	w7 = (points1[:,1]).reshape(points2.shape[0], 1)
	w8 = np.ones(w7.shape)
	W = np.concatenate([w0,w1,w2,w3,w4,w5,w6,w7,w8], 1)
	_, _, V_T = np.linalg.svd(W)
	F11,F12,F13,F21,F22,F23,F31,F32,F33 = V_T[-1]
	F_hat = np.array([[F11,F12,F13], [F21,F22,F23], [F31,F32,F33]])
	U, S, V_T = np.linalg.svd(F_hat)
	S_new = np.array([[S[0],0,0], [0,S[1],0], [0,0,0]])
	F = U.dot(S_new).dot(V_T)
	return F




'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the normalized eight
point algorithm works
'''
def normalized_eight_point_alg(points1, points2):
    # TODO: Implement this method!
	#print 'points1:\n', points1
	centroid1 = np.mean(points1, 0)
	centroid1[2] = 0
	centroid2 = np.mean(points2, 0)
	centroid2[2] = 0
	#print 'centroid 1:', centroid1
	p1 = points1 - centroid1
	p2 = points2 - centroid2
	tx1 = -centroid1[0]
	ty1 = -centroid1[1]
	tx2 = -centroid2[0]
	ty2 = -centroid2[1]
	#print 'p1:', p1
	temp1 = p1 * p1
	temp1 = np.mean(temp1, 0)
	#print 'temp1:', temp1
	temp1 = np.sqrt(2 / temp1)
	temp1[2] = 1
	#print 'temp1:', temp1
	sx1 = temp1[0]
	sy1 = temp1[1]
	p1 *= temp1
	#print 'p1:', p1
	temp2 = p2 * p2
	temp2 = np.mean(temp2, 0)
	temp2 = np.sqrt(2 / temp2)
	temp2[2] = 1
	sx2 = temp2[0]
	sy2 = temp2[1]
	p2 *= temp2
	#print 'p2:', p2
	T1t = np.array([[1,0,tx1], [0,1,ty1], [0,0,1]])
	T1s = np.array([[sx1,0,0], [0,sy1,0], [0,0,1]])
	T1 = T1s.dot(T1t)
	T2t = np.array([[1,0,tx2], [0,1,ty2], [0,0,1]])
	T2s = np.array([[sx2,0,0], [0,sy2,0], [0,0,1]])
	T2 = T2s.dot(T2t)

	#print 'T1 old:', T1old
	#print 'T1:', T1
	#print 'points1[0]:', points1[0]
	#print 'calculation:', T1.dot(points1[0].reshape(3,1)).T
	#print 'p1:', p1
	#print 'calculation2:', T1.dot(points1[-1].reshape(3,1)).T

	Fq = lls_eight_point_alg(p1, p2)
	F = T2.T.dot(Fq).dot(T1)
	#print 'F:', F
	return F



'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image 
    im2 - a HxW(xC) matrix that contains pixel values from the second image 
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):
    # TODO: Implement this method!
	l1 = (F.T.dot(points2.T)).T
	l2 = (F.dot(points1.T)).T
	plt.imshow(im1, cmap=plt.cm.gray)
	for point in points1:
		plt.plot(point[0], point[1], marker='*', markersize=10, color="blue")
	for line in l1:
		x = np.array(range(512))
		a, b, c = line
		y = -a/b*x - c/b
		plt.ylim((511,0))
		plt.plot(x, y, lw=1, color="red")
	plt.show()
	plt.imshow(im2, cmap=plt.cm.gray)
	for point in points2:
		plt.plot(point[0], point[1], marker='*', markersize=10, color="blue")
	for line in l2:
		x = np.array(range(512))
		a, b, c = line
		y = -a/b*x - c/b
		plt.ylim((511,0))
		plt.plot(x, y, lw=1, color="red")
	plt.show()

	


'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
points to their corresponding epipolar lines
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    # TODO: Implement this method!
	l1 = (F.T.dot(points2.T)).T
	d_tot = 0
	for i in range(len(l1)):
		a, b, c = l1[i]
		x = points1[i,0]
		y = points1[i,1]
		d_tot += np.absolute(a*x + b*y + c) / np.sqrt(a**2 + b**2)
	average_distance = d_tot/len(l1)
	return average_distance

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print '-'*80
        print "Set:", im_set
        print '-'*80

        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        F_lls = lls_eight_point_alg(points1, points2)
        print "Fundamental Matrix from LLS  8-point algorithm:\n", F_lls
        print "Distance to lines in image 1 for LLS:", \
            compute_distance_to_epipolar_lines(points1, points2, F_lls)
        print "Distance to lines in image 2 for LLS:", \
            compute_distance_to_epipolar_lines(points2, points1, F_lls.T)

        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points2[i].dot(F_normalized.dot(points1[i])) 
            for i in xrange(points1.shape[0])]
        print "p'^T F p =", np.abs(pFp).max()
        print "Fundamental Matrix from normalized 8-point algorithm:\n", \
            F_normalized
        print "Distance to lines in image 1 for normalized:", \
            compute_distance_to_epipolar_lines(points1, points2, F_normalized)
        print "Distance to lines in image 2 for normalized:", \
            compute_distance_to_epipolar_lines(points2, points1, F_normalized.T)

        # Plotting the epipolar lines
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)

        plt.show()




