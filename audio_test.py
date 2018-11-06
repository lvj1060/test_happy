from pylab import*
from scipy.io import wavfile

sampFreq, snd = wavfile.read('440_sine.wav')
print(snd.dtype)
print(snd.shape)
s1 = snd[:, 0]
timeArray = arange(0, 5292.0, 1)   #[0s, 1s], 5060个点
timeArray = timeArray / sampFreq   #[0s, 0.114s]
timeArray = timeArray * 1000       #[0ms, 114ms]
plot(timeArray, s1, color='k')
ylabel('Amplitude')
xlabel('Time (ms)')
plt.show()