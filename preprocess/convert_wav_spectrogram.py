import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import argparse

def create_spectrogram(wav_file, output_dir):
    clip, sample_rate = librosa.load(wav_file, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

    image_file=os.path.join(output_dir, wav_file.split('/')[-1].split('.wav')[0]+'.png')
    plt.savefig(image_file, dpi=400, bbox_inches='tight',pad_inches=0)

    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del image_file, clip, sample_rate, fig, ax, S

    return

def main(input_dir, output_dir):
    wav_files = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
    for wav_file in wav_files:
        create_spectrogram(wav_file, output_dir)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',
                        required=False,
                        default='/data/datasets/esc50/ESC-50-master/audio/',
                        help='''Directory where .wav files live''')
    parser.add_argument('--output_dir',
                        required=False,
                        default='/data/datasets/esc50/esc50_processed/images',
                        help='''Directory where to save the .png files''')
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir)
