import numpy as np
from typing import Iterable
import pandas as pd
import tensorflow as tf
from scipy import ndimage

class Preprocesser:
    
    def preprocessPipeline(self, 
                           images: Iterable, 
                           basicProcessing: bool = False,
                           randomRotation: bool = False,
                           randomFlip: bool = False,
                           randomZoom: bool = False,
                           randomShear: bool = False,
                           filtering: bool = False):

        for i in range(len(images)):
            image = images[i]

            if basicProcessing:
                image = self.basicProcessing(image)

            if randomRotation:
                image = self.randomRotation(image)

            if randomFlip:
                image = self.randomFlip(image)

            if randomZoom:
                image = self.randomZoom(image)

            if randomShear:
                image = self.randomShear(image)

            if filtering:
                for channel in range(len(image)):
                    image[:, :, channel] = self.anisotropicFiltering2D(image[:, :, channel])

            images[i] = image

        return images

    def basicProcessing(image):
        image = tf.image.adjust_contrast(image, 2)
        image = tf.image.adjust_gamma(image, gamma=3, gain=2)
        return image 

    def randomRotation(image):
        return tf.keras.preprocessing.image.random_rotation(image, 
                                                            180,
                                                            row_axis=0,
                                                            col_axis=1,
                                                            channel_axis=2)

    def randomFlip(image):
        randInts = np.random.randint(0, 2, 2)

        if randInts[0]==1:
            image = tf.image.flip_left_right(image)
        
        if randInts[1]==1:
            image = tf.image.flip_up_down(image)

        return image

    def randomZoom(image):
        return tf.keras.preprocessing.image.random_zoom(image, 
                                                        (0.5, 1.5),
                                                        row_axis=0,
                                                        col_axis=1,
                                                        channel_axis=2)

    def randomShear(image):
        return tf.keras.preprocessing.image.random_shear(image, 
                                                        45,
                                                        row_axis=0,
                                                        col_axis=1,
                                                        channel_axis=2)

    def anisotropicFiltering2D(img,
                               niter=1,
                               kappa=50,
                               gamma=0.1,
                               step=(1.,1.),
                               option=1,
                               ploton=False):
        '''
        Reference: 
        P. Perona and J. Malik. 
        Scale-space and edge detection using ansotropic diffusion.
        IEEE Transactions on Pattern Analysis and Machine Intelligence, 
        12(7):629-639, July 1990.

        Alistair Muldal
        Department of Pharmacology
        University of Oxford
        '''
 
        # initialize output array
        img = img.astype('float32')
        imgout = img.copy()
    
        # initialize some internal variables
        deltaS = np.zeros_like(imgout)
        deltaE = deltaS.copy()
        NS = deltaS.copy()
        EW = deltaS.copy()
        gS = np.ones_like(imgout)
        gE = gS.copy()
    
        # create the plot figure, if requested
        if ploton:
            import pylab as pl
            from time import sleep
    
            fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
            ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)
    
            ax1.imshow(img,interpolation='nearest')
            ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
            ax1.set_title("Original image")
            ax2.set_title("Iteration 0")
    
            fig.canvas.draw()
    
        for ii in range(niter):
    
            # calculate the diffs
            deltaS[:-1,: ] = np.diff(imgout,axis=0)
            deltaE[: ,:-1] = np.diff(imgout,axis=1)
    
            # conduction gradients (only need to compute one per dim!)
            if option == 1:
                gS = np.exp(-(deltaS/kappa)**2.)/step[0]
                gE = np.exp(-(deltaE/kappa)**2.)/step[1]
            elif option == 2:
                gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
                gE = 1./(1.+(deltaE/kappa)**2.)/step[1]
    
            # update matrices
            E = gE*deltaE
            S = gS*deltaS
    
            # subtract a copy that has been shifted 'North/West' by one
            # pixel. don't as questions. just do it. trust me.
            NS[:] = S
            EW[:] = E
            NS[1:,:] -= S[:-1,:]
            EW[:,1:] -= E[:,:-1]
    
            # update the image
            imgout += gamma*(NS+EW)
    
            if ploton:
                iterstring = "Iteration %i" %(ii+1)
                ih.set_data(imgout)
                ax2.set_title(iterstring)
                fig.canvas.draw()
                # sleep(0.01)
    
        return imgout
