import argparse
import os
import skimage.io
import librosa
import numpy as np

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def spectrogram_image(y, sr, hop_length, n_mels):
    #use log mel spectogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=hop_length**2, hop_length=hop_length)
    #add small number to avoid log(0)
    mels = np.log(mels + 1e-9)
    #min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    #put low frequences at the bottom of image
    img = np.flip(img, axis=0)
    #invert, black==more energy
    img = 255-img

    return img

def create_spectogram(wav_file, hop_length, n_mels, time_steps):
    y, sr = librosa.load(wav_file, sr=22050)
    start_sample = 0
    length_samples = time_steps*hop_length
    window = y[start_sample:start_sample+length_samples-1]
    img = spectrogram_image(window, sr, hop_length, n_mels)

    return img

def main(input_dir, hop_length, n_mels, time_steps, output_dir):

    wav_files = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
    for wav_file in wav_files:
        img = create_spectogram(wav_file, hop_length, n_mels, time_steps)
        img_name = os.path.join(output_dir, wav_file.split('/')[-1].split('.wav')[0]+'.png')
        skimage.io.imsave(img_name, img)
        
    return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',
                        required=False,
                        default='/data/datasets/esc50/ESC-50-master/audio/',
                        help='''Directory where .wav files live''')
    parser.add_argument('--hop_length',
                        required=False,
                        default=512,
                        help='''Number of samples per time-step in spectogram''')
    parser.add_argument('--n_mels',
                        required=False,
                        default=1080,
                        help='''Number of bins in spectogram, height of image''')
    parser.add_argument('--time_steps',
                        required=False,
                        default=200,
                        help='''Number of time-steps, width of image''')
    parser.add_argument('--output_dir',
                        required=False,
                        default='/data/datasets/esc50/esc50_processed/images',
                        help='''Directory where to save the .png files''')

    args = parser.parse_args()

    main(input_dir=args.input_dir, hop_length=args.hop_length,
         n_mels=args.n_mels, time_steps=args.time_steps,
         output_dir=args.output_dir)
