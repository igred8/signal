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



def smooth(xvec, yvec, width, mode='gauss', edgemode='valid'):
    """
    xvec
    yvec
    
    width - float. if mode == 'gauss' width=std, if 'step' [center-width/2, center+width/2]
    mode - {'gauss','step', 'median'}
    edgemode = {'valid','pad'}
    to be used with gaussian or stepfunc

    When 'gauss' or 'step' filtering modes are used, both xvec and yvec are convolved with a gaussian or step function. The output x and y vecs are of shorter length because the 'valid' values begin at the point when the span of the convolution function (gauss/step) is inside the boundaries of the original xvec. 
    If the edgemode='valid' is used, then the output xvec and yvec are shorter than the input vectors, which may lead to problems when trying to compare initial to filtered signals. 
    The edgemode='pad' features aims to rectify this by padding the filtered yvec with the edge value on both sides, while keeping the original xvec values there. This method does introduce a discontinuity of the the derivative of the filtered signal.  
    """
    center = xvec.min() + (xvec.max() - xvec.min() ) / 2
    xstep = xvec[1]-xvec[0]
    xlen = xvec.shape[0]

    if mode in ['gauss', 'step']:
        wvec = 0
        if mode == 'gauss':
            nsig = 5
            xvectemp = np.arange(center - nsig * width, center + nsig * width + xstep, xstep)
            wvec = gaussian(xvectemp, center, width)
        elif mode == 'step':
            xvectemp = np.arange(center - width/2, center + width/2 + xstep, xstep)
            wvec = stepfunc(xvectemp, center, width)
        
        cnorm = 1 / (np.sum(wvec))
        # use mode='valid', otherwise edge/boundary values are not intuitive
        yvec = cnorm * np.convolve(yvec, wvec, mode='valid')
        
        ndiff = xlen - yvec.shape[0]
        # if ndiff < 0:
        #     print('WARNING: `width` is larger than `xvec` span.')  
        if edgemode == 'pad':
            
            yvec = np.pad(yvec, [ndiff // 2, ndiff - ndiff // 2], mode='edge')

        elif edgemode == 'valid':
            xvec = cnorm * np.convolve(xvec, wvec, mode='valid') # done to match length of x and y vecs

    elif mode == 'median':
        yvec = sps.medfilt(yvec, kernel_size=width)
    else:
        print("ERROR: _mode_ must be 'gauss', 'step', or 'median' ")
        return 1


    return xvec, yvec

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
    nsamples = sigvec.size
    
    # FFT of signal with appropriate normalization
    sigfft = (1 / nsamples) * np.fft.fft(sigvec, n=nsamples)

    npad = nsamples * (samplerate_increase_factor - 1)

    sfftpad = np.pad(np.fft.fftshift(sigfft), npad//2, mode='constant', constant_values=0)

    sigvec_upsampled = np.fft.ifft(np.fft.ifftshift(sfftpad))
    sigvec_upsampled = np.real_if_close(sigvec_upsampled) # take only the real value if it's close

    return sigvec_upsampled

