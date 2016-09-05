from os import walk
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.nnet.neighbours import images2neibs 
from theano.tensor.nnet.neighbours import neibs2images

'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def plot_mul(c, D, im_num, X_mn, num_coeffs, n_blocks):
    '''
    Plots nine PCA reconstructions of a particular image using number
    of components specified by num_coeffs

    Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the image blocks.
        n represents the maximum dimension of the PCA space.
        m is (number of images x n_blocks**2)

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)

    im_num: Integer
        index of the image to visualize

    X_mn: np.ndarray
        a matrix representing the mean block.

    num_coeffs: Iterable
        an iterable with 9 elements representing the number_of coefficients
        to use for reconstruction for each of the 9 plots

    n_blocks: Integer
        number of blocks comprising the image in each direction.
        For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4
    '''
    f, axarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            nc = num_coeffs[i*3+j]
            cij = c[:nc, n_blocks*n_blocks*im_num:n_blocks*n_blocks*(im_num+1)]
            Dij = D[:, :nc]
            plot(cij, Dij, n_blocks, X_mn, axarr[i, j])

    f.savefig('output/hw1a_{0}_im{1}.png'.format(n_blocks, im_num))
    print 'output/hw1a_{0}_im{1}.png'.format(n_blocks, im_num)
    plt.close(f)

def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image block of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of each block

    imname: string
        name of file where image will be saved.
    '''

    topSixteen = D[:,:16]
    f, axarr = plt.subplots(4, 4)
    for i in range(4):
        for j in range(4):
            topSixteenReshaped = topSixteen[:,i*4+j]
            topSixteenReshaped = np.reshape(topSixteenReshaped, (sz,sz))
            axarr[i][j].imshow(topSixteenReshaped)
    f.savefig(imname)
    plt.close(f)

    '''raise NotImplementedError'''


def plot(c, D, n_blocks, X_mn, ax):
    '''
    Plots a reconstruction of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        n_blocks: Integer
            number of blocks comprising the image in each direction.
            For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4

        X_mn: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        ax: the axis on which the image will be plotted
    '''
    neibs = theano.tensor.matrix('neibs')
    x = np.dot(D,c).T
    x = x + np.repeat(X_mn.reshape(1, -1), x.shape[0], 0)

    im_new = neibs2images(neibs, (256/n_blocks, 256/n_blocks), (1,1,256,256))
    inv_window = theano.function([neibs], im_new)
    im_new_val = inv_window(x)
    im_new_val = np.reshape(im_new_val,(256,256))

    ax.imshow(im_new_val, cmap=cm.Greys_r)



    '''raise NotImplementedError'''


def main():
    '''
    Read here all images(grayscale) from jaffe folder
    into an numpy array Ims with size (no_images, height, width).
    Make sure the images are read after sorting the filenames
    '''
    fnames = []
    for (dirpath, dirnames, filenames) in walk("jaffe/"):
        fnames.extend(filenames)
    fnames.sort()
    Ims = []
    for fname in fnames:
        im = Image.open("jaffe/"+fname)
        im = im.convert("L")
        Ims.append(np.array(im))
    Ims = np.array(Ims)
    

    szs = [16, 32, 64]
    num_coeffs = [range(1, 10, 1), range(3, 30, 3), range(5, 50, 5)]

    for sz, nc in zip(szs, num_coeffs):
        '''
        Divide here each image into non-overlapping blocks of shape (sz, sz).
        Flatten each block and arrange all the blocks in a
        (no_images*n_blocks_in_image) x (sz*sz) matrix called X
        '''
        
        X = None
        for im in Ims:
            images = T.tensor4('images')
            neibs = images2neibs(images, neib_shape=(sz, sz))
            window_function = theano.function([images], neibs)
            im = np.reshape(im, (1,1,Ims.shape[1],Ims.shape[2]))
            neibs_val = window_function(im)
            if X is None:
                X = neibs_val
            else:
                X = np.vstack((X,neibs_val))
        
                
        X_mn = np.mean(X, 0)
        X = X - np.repeat(X_mn.reshape(1, -1), X.shape[0], 0)

        '''
        Perform eigendecomposition on X^T X and arrange the eigenvectors
        in decreasing order of eigenvalues into a matrix D
        '''

        w, v = np.linalg.eig(np.dot(X.T,X))

        idx = w.argsort()[::-1]   
        w = w[idx]
        D = v[:,idx]


        c = np.dot(D.T, X.T)

        for i in range(0, 200, 10):
            plot_mul(c, D, i, X_mn.reshape((sz, sz)),
                     num_coeffs=nc, n_blocks=int(256/sz))


        plot_top_16(D, sz, imname='output/hw1a_top16_{0}.png'.format(sz))


if __name__ == '__main__':
    main()
