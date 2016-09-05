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

def plot_mul(c, D, im_num, X_mn, num_coeffs):
    '''
    Plots nine PCA reconstructions of a particular image using number
    of components specified by num_coeffs

    Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the images
        n represents the maximum dimension of the PCA space.
        m represents the number of images

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in the image)

    im_num: Integer
        index of the image to visualize

    X_mn: np.ndarray
        a matrix representing the mean image
    '''
    f, axarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            nc = num_coeffs[i*3+j]
            cij = c[:nc, im_num]
            Dij = D[:, :nc]
            plot(cij, Dij, X_mn, axarr[i, j])

    f.savefig('output/hw1b_im{0}.png'.format(im_num))
    plt.close(f)


def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in the image)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of a image

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
    # raise NotImplementedError


def plot(c, D, X_mn, ax):
    '''
    Plots a reconstruction of a particular image using D as the basis matrix and c as
    the coefficient vector
    Parameters
    -------------------
        c: np.ndarray
            a l x 1 vector  representing the coefficients of the image.
            l represents the dimension of the PCA space used for reconstruction

        D: np.ndarray
            an N x l matrix representing first l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in the image)

        X_mn: basis vectors represent the divergence from the mean so this
            matrix should be added to the reconstructed image

        ax: the axis on which the image will be plotted
    '''
    x = np.dot(D,c).T
    x = x.reshape((256,256)) + X_mn

    # im_new = neibs2images(neibs, (256, 256), (1,1,256,256))
    # inv_window = theano.function([neibs], im_new,allow_input_downcast=True)
    # im_new_val = inv_window(x)
    # im_new_val = np.reshape(im_new_val,(256,256))

    ax.imshow(x, cmap=cm.Greys_r)
    '''raise NotImplementedError'''


if __name__ == '__main__':
    '''
    Read all images(grayscale) from jaffe folder and collapse each image
    to get an numpy array Ims with size (no_images, height*width).
    Make sure to sort the filenames before reading the images
    '''

    fnames = []
    for (dirpath, dirnames, filenames) in walk("jaffe/"):
        fnames.extend(filenames)
    fnames.sort()
    Ims = None
    for fname in fnames:
        im = Image.open("jaffe/"+fname)
        im = im.convert("L")
        im = np.array(im)
        im = np.reshape(im,(1,im.shape[0]*im.shape[1]))
        if Ims is None:
            Ims = im
        else:
            Ims = np.vstack((Ims,im))

    print Ims.shape

    Ims = Ims.astype(np.float32)
    X_mn = np.mean(Ims, 0)
    X = Ims - np.repeat(X_mn.reshape(1, -1), Ims.shape[0], 0)

    '''
    Use theano to perform gradient descent to get top 16 PCA components of X
    Put them into a matrix D with decreasing order of eigenvalues

    If you are not using the provided AMI and get an error "Cannot construct a ufunc with more than 32 operands" :
    You need to perform a patch to theano from this pull(https://github.com/Theano/Theano/pull/3532)
    Alternatively you can downgrade numpy to 1.9.3, scipy to 0.15.1, matplotlib to 1.4.2
    '''
    N = 16
    eta = 0.0001
    max_step = 100
    D = np.random.rand(65536,N)
    lambdaI = []
    for i in range(0,16):
        step = 0
        print i
        while step < max_step:
            tempSum = 0
            for j in range(i):
                tempProd = np.dot(D[:,j].T,D[:,i])
                tempProd = np.dot(D[:,j],tempProd)
                tempSum = tempSum + lambdaI[j]*tempProd
            grad = 2*(np.dot(X.T,np.dot(X,D[:,i])) - tempSum)
            gd = D[:,i] + eta*grad
            D[:,i] = gd/np.linalg.norm(gd)
            step += 1
        lambdaI.append(np.dot(X.T,np.dot(X,D[:,i])))

    c = np.dot(D.T, X.T)

    for i in range(0,200,10):
        plot_mul(c, D, i, X_mn.reshape((256, 256)),
                 [1, 2, 4, 6, 8, 10, 12, 14, 16])

    plot_top_16(D, 256, 'output/hw1b_top16_256.png')

