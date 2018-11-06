import numpy
import numpy as np
import scipy.io.wavfile
from matplotlib import pyplot as plt
from scipy.fftpack import dct

def MFCC(wav_name):
    sample_rate,signal=scipy.io.wavfile.read(wav_name)

    print(sample_rate,len(signal))
    #读取前3.5s 的数据
    signal=signal[0:int(4*sample_rate)]

    plt.plot(signal)
    # plt.show()

    f1 = plt.figure()
    pre_emphasis = 0.97
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    ax2=f1.add_subplot(2,2,2)
    ax2.plot(emphasized_signal)
    # plt.show()


    frame_size=0.025
    frame_stride=0.01
    frame_length,frame_step=frame_size*sample_rate,frame_stride*sample_rate
    signal_length=len(emphasized_signal)
    frame_length=int(round(frame_length))
    frame_step=int(round(frame_step))
    num_frames=int(numpy.ceil(float(numpy.abs(signal_length-frame_length))/frame_step))


    pad_signal_length=num_frames*frame_step+frame_length
    z=numpy.zeros((pad_signal_length-signal_length))
    pad_signal=numpy.append(emphasized_signal,z)

    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

    frames = pad_signal[numpy.mat(indices).astype(numpy.int32, copy=False)]

    ax3=f1.add_subplot(2,2,3)
    ax3.plot(pad_signal)
    # plt.show()


    NFFT = 512
    frames *= numpy.hamming(frame_length)
    # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **
    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    low_freq_mel = 0
    #将频率转换为Mel
    nfilt = 40
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz

    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    f2 = plt.figure()
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB

    num_ceps = 20
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
    (nframes, ncoeff) = mfcc.shape

    n = numpy.arange(ncoeff)
    cep_lifter =22
    lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
    mfcc *= lift  #*
    filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
    plt.imshow(numpy.flipud(filter_banks.T), cmap=plt.cm.jet, aspect=0.2, extent=[0,filter_banks.shape[1],0,filter_banks.shape[0]]) #画热力图
    # plt.show()
    mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)
    a=mfcc
    print(mfcc)

    # arr= np.array(mfcc)
    # str=' '.join(arr)
    # print(str)
    np.savetxt("a.txt", mfcc, fmt="%.3f", delimiter=",")


    # fileObject = open('sampleList.txt', 'wb')
    # for ip in mfcc:
    #     fileObject.write(ip)
    #     # fileObject.write('\n')
    # fileObject.close()
    plt.imshow(numpy.flipud(mfcc.T), cmap=plt.cm.jet, aspect=0.2, extent=[0,mfcc.shape[0],0,mfcc.shape[1]])#热力图
    # plt.show()
    print("-------------------------------------------------------------------")
    print(mfcc.shape)
    return mfcc

MFCC('break2.wav')