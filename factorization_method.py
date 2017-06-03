import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.gridspec as gridspec
from epipolar_utils import *

'''
FACTORIZATION_METHOD The Tomasi and Kanade Factorization Method to determine
the 3D structure of the scene and the motion of the cameras.
Arguments:
    points_im1 - N points in the first image that match with points_im2
    points_im2 - N points in the second image that match with points_im1

    Both points_im1 and points_im2 are from the get_data_from_txt_file() method
Returns:
    structure - the structure matrix
    motion - the motion matrix
'''
def factorization_method(points_im1, points_im2):
    # TODO: Implement this method!
	centroid1 = np.mean(points_im1, axis=0)
	centroid2 = np.mean(points_im2, axis=0)
	im1 = points_im1 - centroid1
	im2 = points_im2 - centroid2
	#print 'centroids:', centroid1, centroid2, centroid2.shape
	#print 'im1 im2:\n', im1, im2
	im1 = im1.T[:2]
	im2 = im2.T[:2]
	D = np.concatenate([im1, im2], 0)
	#print 'points_im1:\n', im1, im1.shape
	#print 'points_im2:\n', im2, im2.shape
	#print 'D:\n', D, D.shape
	U, W, V_T = np.linalg.svd(D)
	#print 'SVD:', U.shape, W.shape, V_T.shape
	print 'W:', W
	W3 = np.array([[W[0],0,0], [0,W[1],0], [0,0,W[2]]])
	M = U[:,:3].dot(np.sqrt(W3))
	#print 'V_T:', V_T[:3].shape
	S = np.sqrt(W3).dot(V_T[:3])
	#print 'M, S:', M.shape, S.shape
	return S, M
	


if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set1_subset']:
        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points_im1 = get_data_from_txt_file(im_set + '/pt_2D_1.txt')
        points_im2 = get_data_from_txt_file(im_set + '/pt_2D_2.txt')
        points_3d = get_data_from_txt_file(im_set + '/pt_3D.txt')
        assert (points_im1.shape == points_im2.shape)

        # Run the Factorization Method
        structure, motion = factorization_method(points_im1, points_im2)

        # Plot the structure
        fig = plt.figure()
        ax = fig.add_subplot(121, projection = '3d')
        scatter_3D_axis_equal(structure[0,:], structure[1,:], structure[2,:], ax)
        ax.set_title('Factorization Method')
        ax = fig.add_subplot(122, projection = '3d')
        scatter_3D_axis_equal(points_3d[:,0], points_3d[:,1], points_3d[:,2], ax)
        ax.set_title('Ground Truth')

        plt.show()


