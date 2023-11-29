# MIT License
#
# Copyright (c) 2019 Michele Maione, mikymaione@hotmail.it
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# https://github.com/mikymaione/2SRPy

import os
import cv2
import numpy


class SkinColorFilter():
    """
    This class implements a number of functions to perform skin color filtering.
    It is based on the work published in "Adaptive skin segmentation via feature-based face detection",
    M.J. Taylor and T. Morris, Proc SPIE Photonics Europe, 2014 [taylor-spie-2014]_
    Attributes
    ----------
    mean: numpy.ndarray | dim: 2
        the mean skin color
    covariance: numpy.ndarray | dim: 2x2
        the covariance matrix of the skin color
    covariance_inverse: numpy.ndarray | dim: 2x2
        the inverse covariance matrix of the skin color
    circular_mask: numpy.ndarray
        mask of the size of the image, defining a circular region in the center
    luma_mask: numpy.ndarray
        mask of the size of the image, defining valid luma values
    """

    def __init__(self):
        self.mean = numpy.array([0.0, 0.0])
        self.covariance = numpy.zeros((2, 2), 'float64')
        self.covariance_inverse = numpy.zeros((2, 2), 'float64')

    def __generate_circular_mask(self, image, radius_ratio=0.4):
        """
        This function will generate a circular mask to be applied to the image.
        The mask will be true for the pixels contained in a circle centered in the image center, and with radius equals to radius_ratio * the image's height.
        Parameters
        ----------
        image: numpy.ndarray
            The face image.
        radius_ratio: float:
            The ratio of the image's height to define the radius of the circular region. Defaults to 0.4.
        """
        w = image.shape[0]
        h = image.shape[1]
        x_center = w / 2
        y_center = h / 2
        # arrays with the image coordinates
        X = numpy.zeros((w, h))
        X[:] = range(0, w)
        Y = numpy.zeros((h, w))
        Y[:] = range(0, h)
        Y = numpy.transpose(Y)
        # translate s.t. the center is the origin
        X -= x_center
        Y -= y_center
        # condition to be inside of a circle: x^2 + y^2 < r^2
        radius = radius_ratio * h
        # x ^ 2 + y ^ 2 < r ^ 2
        cm = (X ** 2 + Y ** 2) < (radius ** 2)  # dim : w x h
        self.circular_mask = cm

    def __remove_luma(self, image):
        """
        This function remove pixels with extreme luma values.
        Some pixels are considered as non-skin if their intensity is either too high or too low.
        The luma value for all pixels inside a provided circular mask is calculated. Pixels for which the luma value deviates more than 1.5 * standard deviation are pruned.
        Parameters
        ----------
        image: numpy.ndarray
            The face image.
        """
        # compute the mean and std of luma values on non-masked pixels only
        R = 0.299 * image[self.circular_mask, 0]
        G = 0.587 * image[self.circular_mask, 1]
        B = 0.114 * image[self.circular_mask, 2]
        luma = R + G + B
        m = numpy.mean(luma)
        s = numpy.std(luma)
        # apply the filtering to the whole image to get the luma mask
        R = 0.299 * image[:, :, 0]
        G = 0.587 * image[:, :, 1]
        B = 0.114 * image[:, :, 2]
        luma = R + G + B
        # dim : image.x x image.y
        lm = numpy.logical_and((luma > (m - 1.5 * s)), (luma < (m + 1.5 * s)))
        self.luma_mask = lm

    def __RG_Mask(self, image, dtype=None):
        # dim: image.x x image.y
        channel_sum = image[:, :, 0].astype('float64') + image[:, :, 1] + image[:, :, 2]
        # dim: image.x x image.y
        nonzero_mask = numpy.logical_or(numpy.logical_or(image[:, :, 0] > 0, image[:, :, 1] > 0), image[:, :, 2] > 0)
        # dim: image.x x image.y
        R = numpy.zeros((image.shape[0], image.shape[1]), dtype)
        R[nonzero_mask] = image[nonzero_mask, 0] / channel_sum[nonzero_mask]
        # dim: image.x x image.y
        G = numpy.zeros((image.shape[0], image.shape[1]), dtype)
        G[nonzero_mask] = image[nonzero_mask, 1] / channel_sum[nonzero_mask]
        return R, G

    def estimate_gaussian_parameters(self, image):
        """
        This function estimates the parameter of the skin color distribution.
        The mean and covariance matrix of the skin pixels in the normalised rg colorspace are computed.
        Note that only the pixels for which both the circular and the luma mask is 'True' are considered.
        Parameters
        ----------
        image: numpy.ndarray
            The face image.
        """
        self.__generate_circular_mask(image)
        self.__remove_luma(image)
        # dim: image.x x image.y
        mask = numpy.logical_and(self.luma_mask, self.circular_mask)
        # get the mean
        # R dim: image.x x image.y
        # G dim: image.x x image.y
        R, G = self.__RG_Mask(image)
        # dim: 2
        self.mean = numpy.array([numpy.mean(R[mask]), numpy.mean(G[mask])])
        # get the covariance
        R_minus_mean = R[mask] - self.mean[0]
        G_minus_mean = G[mask] - self.mean[1]
        samples = numpy.vstack((R_minus_mean, G_minus_mean))
        samples = samples.T
        cov = sum([numpy.outer(s, s) for s in samples])  # dim: 2x2
        self.covariance = cov / float(samples.shape[0] - 1)
        # store the inverse covariance matrix (no need to recompute)
        if numpy.linalg.det(self.covariance) != 0:
            self.covariance_inverse = numpy.linalg.inv(self.covariance)
        else:
            self.covariance_inverse = numpy.zeros_like(self.covariance)

    def get_skin_mask(self, image, threshold=0.5):
        """
        This function computes the probability of skin-color for each pixel in the image.
        Parameters
        ----------
        image: numpy.ndarray
            The face image.
        threshold: float: 0->1
            The threshold on the skin color probability. Defaults to 0.5
        Returns
        -------
        skin_mask: numpy.ndarray
            The mask where skin color pixels are labeled as True.
        """
        # get the image in rg colorspace
        R, G = self.__RG_Mask(image, 'float64')
        # compute the skin probability map
        R_minus_mean = R - self.mean[0]
        G_minus_mean = G - self.mean[1]
        n = R.shape[0] * R.shape[1]
        V = numpy.dstack((R_minus_mean, G_minus_mean))  # dim: image.x x image.y
        V = V.reshape((n, 2))  # dim: nx2
        probs = [numpy.dot(k, numpy.dot(self.covariance_inverse, k)) for k in V]
        probs = numpy.array(probs).reshape(R.shape)  # dim: image.x x image.y
        skin_map = numpy.exp(-0.5 * probs)  # dim: image.x x image.y
        return skin_map > threshold


class SSR():
    """
    This class implements the Spatial Subspace Rotation for Remote Photoplethysmography

    It is based on the work published in "A Novel Algorithm for Remote Photoplethysmography - Spatial Subspace Rotation",
    Wenjin Wang, Sander Stuijk, and Gerard de Haan, IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. 63, NO. 9, SEPTEMBER 2016
    """

    def calulate_pulse_signal(self, images, show, fps):
        """
        Parameters
        ----------
        images: List<numpy.ndarray | dim: HxWx3>
            The images to elaborate

        show: int [0/1]
            Show the plot

        fps: int
            Frame per seconds

        Returns
        -------
        k : int
            The number of frame elaborated

        P: numpy.ndarray | dim: K = len(images)
            The pulse signal
        """

        k = 0  # the number of frame elaborated
        K = len(images)  # the number of frame to elaborate
        l = fps  # The temporal stride to use

        # the pulse signal
        P = numpy.zeros(K)  # 1 | dim: K

        # store the eigenvalues Λ and the eigenvectors U at each frame
        Λ = numpy.zeros((3, K), dtype='float64')  # dim: 3xK
        U = numpy.zeros((3, 3, K), dtype='float64')  # dim: 3x3xK

        # a class to perform skin color filtering based on the work published in "Adaptive skin segmentation via feature-based face detection".
        skin_filter = SkinColorFilter()

        for i in range(K):  # 2
            face = images[i]  # dim: HxWx3

            # get: skin pixels
            V = self.__get_skin_pixels(skin_filter, face, show, k == 0)  # 3 | dim: (W×H)x3

            # build the correlation matrix
            C = self.__build_correlation_matrix(V)  # 3 | dim: 3x3

            # get: eigenvalues Λ, eigenvectors U
            Λ[:, k], U[:, :, k] = self.__eigs(C)  # 4 | dim Λ: 3 | dim U: 3x3

            # build p and add it to the pulse signal P
            if k >= l:  # 5
                τ = k - l  # 5
                p = self.__build_p(τ, k, l, U, Λ)  # 6, 7, 8, 9, 10, 11 | dim: l
                P[τ:k] += p  # 11

            k = k + 1

        return k, P

    def __build_correlation_matrix(self, V):
        # V dim: (W×H)x3
        V_T = V.T  # dim: 3x(W×H)

        N = V.shape[0]

        # build the correlation matrix
        C = numpy.dot(V_T, V)  # dim: 3x3
        C = C / N

        return C

    def __get_skin_pixels(self, skin_filter, face, show, do_skininit):
        """
        get eigenvalues and eigenvectors, sort them.

        Parameters
        ----------
        C: numpy.ndarray
            The RGB values of skin-colored pixels.

        Returns
        -------
        Λ: numpy.ndarray
            The eigenvalues of the correlation matrix

        U: numpy.ndarray
            The (sorted) eigenvectors of the correlation matrix
        """

        if do_skininit:
            skin_filter.estimate_gaussian_parameters(face)

        skin_mask = skin_filter.get_skin_mask(face)  # dim: wxh

        V = face[skin_mask]  # dim: (w×h)x3
        V = V.astype('float64') / 255.0

        if show:
            # show the skin in the image along with the mask
            cv2.imshow("Image", numpy.hstack([face]))

            if cv2.waitKey(20) & 0xFF == ord('q'):
                return

        return V

    def __eigs(self, C):
        """
        get eigenvalues and eigenvectors, sort them.

        Parameters
        ----------
        C: numpy.ndarray
            The RGB values of skin-colored pixels.

        Returns
        -------
        Λ: numpy.ndarray
            The eigenvalues of the correlation matrix

        U: numpy.ndarray
            The (sorted) eigenvectors of the correlation matrix
        """

        # get eigenvectors and sort them according to eigenvalues (largest first)
        Λ, U = numpy.linalg.eig(C)  # dim Λ: 3 | dim U: 3x3

        idx = Λ.argsort()  # dim: 3x1
        idx = idx[::-1]  # dim: 1x3

        Λ_ = Λ[idx]  # dim: 3
        U_ = U[:, idx]  # dim: 3x3

        return Λ_, U_

    def __build_p(self, τ, k, l, U, Λ):
        """
        builds P

        Parameters
        ----------
        k: int
            The frame index

        l: int
            The temporal stride to use

        U: numpy.ndarray
            The eigenvectors of the c matrix (for all frames up to counter).

        Λ: numpy.ndarray
            The eigenvalues of the c matrix (for all frames up to counter).

        Returns
        -------
        p: numpy.ndarray
            The p signal to add to the pulse.
        """

        # SR'
        SR = numpy.zeros((3, l), 'float64')  # dim: 3xl
        z = 0

        for t in range(τ, k, 1):  # 6, 7
            a = Λ[0, t]
            b = Λ[1, τ]
            c = Λ[2, τ]
            d = U[:, 0, t].T
            e = U[:, 1, τ]
            f = U[:, 2, τ]
            g = U[:, 1, τ].T
            h = U[:, 2, τ].T

            x1 = a / b
            x2 = a / c
            x3 = numpy.outer(e, g)
            x4 = numpy.dot(d, x3)
            x5 = numpy.outer(f, h)
            x6 = numpy.dot(d, x5)
            x7 = numpy.sqrt(x1)
            x8 = numpy.sqrt(x2)
            x9 = x7 * x4
            x10 = x8 * x6
            x11 = x9 + x10

            SR[:, z] = x11  # 8 | dim: 3
            z += 1

        # build p and add it to the final pulse signal
        s0 = SR[0, :]  # dim: l
        s1 = SR[1, :]  # dim: l

        p = s0 - ((numpy.std(s0) / numpy.std(s1)) * s1)  # 10 | dim: l
        p = p - numpy.mean(p)  # 11

        return p  # dim: l
