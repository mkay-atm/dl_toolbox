import numpy as np
from scipy.linalg import svdvals


def check_if_db(x):
    '''
    check_if_db(X)
        Static method for checking if the input is in dB
        Parameters
        ----------
        x : array or scalar
            representing the signal to noise of a lidar signal.

        Returns
        -------
        bool
            stating wether the input "likely" in dB.

        Notes
        -----
        The method is only tested empirically and therefore not absolute.
    '''
    return np.any(x<-1)


def filter_by_snr(x,snr,snr_threshold):
    '''
    filter_by_snr(X,SNR,SNR_threshold)
        Masking an n-dimensional array (X) according to a given signal to noise ratio (SNR) and specified threshold (SNR_threshold).

        Parameters
        ----------
        x : array_like (intended for using numpy array)
            n-dimensional array representing a signal.
        snr : array_like (intended for using numpy array)
            n-dimensional array of the same dimension as x representing the signal to noise ratio.
        snr_threshold : scalar value, i.e. 0-dimensional
            scalar value giving the lower bounded threshold of the signal to noise threshold.

        order : {'x', 'snr', 'snr_threshold'}

        Returns
        -------
        masked_array, i.e. [data, mask]
            Masked numpy array to be used in further processing.

        Dependencies
        ------------
        functions : check_if_db(x), in_mag(snr)

        Notes
        -----
        If the input SNR is already given in dB, do NOT filter the input SNR in advance for missing values.
        Translation between dB and magnitude can be done with the functions "in_dB(x)" and "in_mag(x)".
        This functions uses machine epsilon (np.float16) for the numerical value of 0.
    '''

    if check_if_db(snr)==True:
        print('SNR interpreted as dB')
        print(snr.min(),snr.max())
        snr= in_mag(snr)
    if check_if_db(snr_threshold)==True:
        print('SNR-threshold interpreted as dB')
        snr_threshold= in_mag(snr_threshold)
    snr_threshold+= np.finfo(np.float32).eps
    return np.ma.masked_where(~(snr>snr_threshold), x)


def in_db(x):
    '''
    in_db(X)
        Calculates dB values of a given input (X). The intended input is the signal to noise ratio of a Doppler lidar.

        Parameters
        ----------
        x : array_like (intended for using numpy array) OR numerical scalar
            n-dimensional array

        Returns
        -------
        X in dB

        Dependencies
        ------------
        functions : check_if_db(x)

        Notes
        -----
        If the input X is already given in dB, X is returned without further processing.
        Please, do NOT filter the input in advance for missing values.
        This functions uses machine epsilon (np.float32) for the numerical value of 0.
    '''

    if check_if_db(x)==True:
        print('Input already in dB')
        return x
    else:
        epsilon_val=  np.finfo(np.float32).eps
        if np.ma.size(x)==0:
            print('0-dimensional input!')
        else:
            if np.ma.size(x)>1:
                x[x<=0]= epsilon_val
                return 10*np.log10(np.ma.masked_where((x<= epsilon_val), x)).filled(10*np.log10(epsilon_val))
            else:
                if x<=0:
                    x= epsilon_val
                return 10*np.log10(np.ma.masked_where((x<= epsilon_val), x)).filled(10*np.log10(epsilon_val))


def in_mag(x):
    '''
    in_mag(X)
        Calculates the magnitude values of a given dB input (X). The intended input is the signal to noise ratio of a Doppler lidar.

        Parameters
        ----------
        x : array_like (intended for using numpy array) OR numerical scalar
            n-dimensional array

        Returns
        -------
        X in magnitude

        Dependencies
        ------------
        functions : check_if_db(x)

        Notes
        -----
        If the input X is already given in magnitde, X is returned without further processing.
        Please, do NOT filter the input in advance for missing values.
        This functions uses machine epsilon (np.float32) for the numerical value of 0.
    '''
    if check_if_db(x)==False:
        print('Input already in magnitude')
        return x
    else:
        epsilon_val=  np.finfo(np.float32).eps
        if np.ma.size(x)==0:
            print('0-dimensional input!')
        else:
            if np.ma.size(x)>1:
                res= 10**(x/10)
                res[res<epsilon_val]= epsilon_val
                return res
            else:
                res= 10**(x/10)
                if res<=epsilon_val:
                    res= epsilon_val
                return res


def CN_est(X):
    Fill_Val = 0
    X_f = X.filled(Fill_Val)
    if np.all(X_f == 0):
        return np.inf
    else:
        max_val = svdvals(X_f).max()
        min_val = svdvals(X_f).min()
        if min_val == 0:
            return np.inf
        else:
            return max_val/min_val
