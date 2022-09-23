# ==========
# Created by Ivan Gadjev
# 2020.02.01
#
# Library of custom functions that aid in projects using pyRadia, genesis, elegant, and varius other codes and analyses. 
#   
# 
# ==========

# standard
# import sys
# import os
from random import gauss
import time
import bisect
import json

# env specific
import scipy.constants as pc
import numpy as np
import scipy.signal as sps
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt

# ==========

def boxfunc(xvec, center, width):
    '''
    Returns the box function evaluated at the values of xvec.
    xvec - ndarray of x values
    center - float. the center of the box
    width - float. the full width of the box

    Uses the sum of two Heaviside functions to make a box.
    H(x+width/2) + H(-x+width/2)
    '''
    midval = 0.5
    boxvec = np.heaviside((xvec - center) + width/2, midval) + np.heaviside( -(xvec - center) + width/2, midval)
    boxvec = boxvec - 1
    return boxvec

def gaussianfunc(xvec, center, width):
    '''
    Returns a Gaussian evaluated at the values of xvec.
    xvec - ndarray of x values
    center - float. the center of the Gaussian
    width - float. full width at half max (FWHM) of the Gaussian 

    Normalized to unity at x=0
    '''

    sigma =  width / (2 * np.sqrt(2 * np.log(2))) # convert FWHM to sigma

    gaussvec = np.exp( (-(xvec - center)**2) / (2*sigma**2) )

    return gaussvec


def smooth(timevec, sigvec, width, mode='gauss', edgemode='valid'):
    """
    timevec - ndarray. time vector
    sigvec - ndarray. signal vector
    
    width - float. if mode == 'gauss' width=FWHM, if 'box' [center-width/2, center+width/2]
    mode - {'gauss','box', 'median'}
    edgemode = {'valid','pad'}
    to be used with gaussianfunc or boxfunc

    When 'gauss' or 'box' filtering modes are used, both timevec and sigvec are convolved with a gaussian or box function. The output x and y vectors are of shorter length because the 'valid' values begin at the point when the span of the convolution function is inside the boundaries of the original timevec. 
    If the edgemode='valid' is used, then the output timevec and sigvec are shorter than the input vectors, which may lead to problems when trying to compare initial to filtered signals. 
    The edgemode='pad' feature aims to rectify this by padding the filtered sigvec with the edge value on both sides, while keeping the original timevec values there. This method does introduce a discontinuity of the the derivative of the filtered signal.  
    """
    center = timevec.min() + (timevec.max() - timevec.min() ) / 2
    xstep = timevec[1]-timevec[0]
    xlen = timevec.shape[0]

    if mode in ['gauss', 'box']:
        wvec = 0
        if mode == 'gauss':
            nsig = 5
            sigma =  width / (2 * np.sqrt(2 * np.log(2))) # convert FWHM to sigma
            timevectemp = np.arange(center - nsig * sigma, center + nsig * sigma + xstep, xstep)
            wvec = gaussianfunc(timevectemp, center, width)

        elif mode == 'box':
            timevectemp = np.arange(center - width/2, center + width/2 + xstep, xstep)
            wvec = boxfunc(timevectemp, center, width)
        
        normconst = 1 / (np.sum(wvec))
        # use mode='valid', otherwise edge/boundary values are not intuitive
        sigvec = normconst * np.convolve(sigvec, wvec, mode='valid')
        
        ndiff = xlen - sigvec.shape[0]
        # if ndiff < 0:
        #     print('WARNING: `width` is larger than `timevec` span.')  
        if edgemode == 'pad':
            
            sigvec = np.pad(sigvec, [ndiff // 2, ndiff - ndiff // 2], mode='edge')

        elif edgemode == 'valid':
            timevec = normconst * np.convolve(timevec, wvec, mode='valid') # done to match length of x and y vecs

    elif mode == 'median':
        sigvec = sps.medfilt(sigvec, kernel_size=width)
    else:
        print("ERROR: _mode_ must be 'gauss', 'box', or 'median' ")
        return 1


    return timevec, sigvec

def fft_scale_phase(timevec, sigvec, convertfactor, *,
mode='weight', power=2, freqlims='auto',
phase=0.0):
    """ Given a time-series signal 
    1. rescale its time axis, based on its main FFT frequency and the given conversion factor 
    2. phase the signal to the cosine-like main frequency oscillation.

    These actions effectively calibrate the time axis to a known length scale, given by convertfactor.
    "Phasing" the signal is based on maximizing the correlation of the signal with a wave of its main frequency for a specified phase. 
    This is a form of peak alignment. Returns the time-shift needed to do this.

    inputs:
    timevec - ndarray(n,)
    sigvec - ndarray(n,)
    convertfactor - float

    mode = {'weight', 'max'}, takes a weighted sum of the frequencies to find the main freq content. 'max' takes the freq with maximum amplitude.
    power = 1, the power for the weighting function. not used for 'max' mode.

    freqlims='auto, if 'auto', then use (0,inf) interval. else use (fmin, fmax) interval. 

    phase - float. radians. the phase of the main freq oscillation. 0 is cosine-like

    returns:
    timescale - scaling factor to calibrate time axis
    timeshift - shift in time for time-align 
    freq_main - main frequency component of the signal
    
    sigfft_freq - frequency vector from FFT
    sigfft - FFT of signal

    """

    # number of samples
    nsamples = sigvec.size
    # zero-pad signal to nearest larger power of 2
    nlen2 = int(2**(np.ceil(np.log2(nsamples))) )
    # FFT of signal with appropriate normalization
    sigfft = (1 / nlen2) * np.fft.fft(sigvec, n=nlen2)
    
    # time step
    timestep = (timevec[1:] - timevec[:-1]).mean()

    # frequency vector
    sigfft_freq = np.fft.fftfreq((nlen2), d=timestep)
    if freqlims == 'auto':
        # take only positive frequencies and exclude zero
        fmin = 0.0
        indlogic = (sigfft_freq > fmin)
    else:
        fmin = freqlims[0]
        fmax = freqlims[1]
        indlogic = (sigfft_freq > fmin) & (sigfft_freq < fmax)
    
    if mode == 'weight':
        # find the main frequency, based on the largest FFT amplitudes averaged with weigths
        # weight of freq is a power of their FFT amplitude

        # inside freqlims 
        sigfft_bound = sigfft[indlogic]
        sigfft_freq_bound = sigfft_freq[indlogic]

        # number of frequencies to average over
        nfreq = np.sum(indlogic)
        
        # normalization constant of frequencies
        normconst =  1 / np.sum(np.abs(sigfft_bound)**power)
        # main frequency
        freq_main = normconst * np.sum(sigfft_freq_bound * np.abs(sigfft_bound)**power)
    
    elif mode == 'max':
        # inside freqlims 
        sigfft_bound = sigfft[indlogic]
        sigfft_freq_bound = sigfft_freq[indlogic]

        #  freq with largest amplitude
        freq_main = sigfft_freq_bound[np.abs(sigfft_bound).argmax()]

    else:
        print('ERROR! Mode not recognized. Please use {"weight", "max"}.')
        return 1


    # scale wavelength to period of undulator
    timescale = convertfactor * freq_main

    # time-align
    
    # create wave with same sampling as signal
    omega = 2*pc.pi*freq_main
    wave = np.cos(omega*timevec + phase)
    # correlate (order of input vectors matters for timeshift sign.)
    tcorr = np.correlate(sigvec, wave, mode='full')
    # convert to a shift in time
    timeshift = (nsamples - (tcorr.argmax() + 1)) * timestep
    
    # mod to 2pi and shift to inside [-pi,pi]
    timeshift = (1/omega) * (np.mod(omega*timeshift - pc.pi, 2*pc.pi) - pc.pi)


    return timescale, timeshift, freq_main, sigfft_freq, sigfft

def align(sig1, sig2):
    """ 
    ---
    EXPERIMENTAL
    ---
    
    Aligns the signals to maximize their correlation. 
    Uses np.correlate().
    sig1 - ndarray
    sig2 - ndarray

    returns: indexshift - this is the index shift for the time vector for the sig1. 
    """
    # correlation
    tcorr = np.correlate(sig1, sig2, mode='full')
    nsamples = sig1.shape[0]
    # shift in index for sig1
    indexshift = (nsamples - (tcorr.argmax() + 1))

    return indexshift, tcorr

def fft_resample(sigvec, samplerate_increase_factor ):
    '''
    Pads the FFT of the signal with zeros at higher frequencies. Returns the inverse FFT which will have a higher sampling rate.

    '''
    
    # number of samples
    nsamples = sigvec.shape[0]
    
    # FFT of signal with appropriate normalization
    sigfft = (1 / nsamples) * np.fft.fft(sigvec, n=nsamples)

    npad = nsamples * (samplerate_increase_factor - 1)

    sfftpad = np.pad(np.fft.fftshift(sigfft), npad//2, mode='constant', constant_values=0)

    sigvec_upsampled = np.fft.ifft(np.fft.ifftshift(sfftpad))
    sigvec_upsampled = np.real_if_close(sigvec_upsampled) # take only the real value if it's close

    return sigvec_upsampled.shape[0]*sigvec_upsampled

