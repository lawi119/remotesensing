import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

sample_wave_file = '/data/ESC-50-master/audio/5-9032-A-0.wav'
sample_rate, samples = wavfile.read(sample_wave_file)
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

sample_wave_file = '/data/ESC-50-master/audio/5-9032-A-0.wav'
sample_rate, samples = wavfile.read(sample_wave_file)
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
