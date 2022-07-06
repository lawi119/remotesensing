import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import argparse
import os

def create_spectrogram(wav_file, window, hop_length, time_steps, output_dir):
    y, sample_rate = librosa.load(wav_file, sr=22050)   #default value
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    if window:
        start_sample = 0
        length_samples = time_steps*hop_length
        y = y[start_sample:start_sample+length_samples-1]

    S = librosa.feature.melspectrogram(y=y, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    
    plt.savefig(output_filename, dpi=400, bbox_inches='tight',pad_inches=0)

    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del image_file, y, sample_rate, fig, ax, S

    return

def define_image_filepath(wav_file, df, output_dir):
    image_filename = wav_file.split('/')[-1].split('.wav')[0]+'.png'
    image_row = df[df['filename']==image_filename]
    object_class = image_row.iloc[2]
    subset = image.row.iloc[-1]
    image_filepath = os.path.join(output_dir, subset, object_class, image_filename)

    return image_filepath
    
def main(input_dir, window, hop_length, time_steps, output_dir):
    
    #preprocess meta file
    df = pd.read_csv('/data/datasets/esc50/ESC-50-master/meta/esc50.csv')
    df['filename'] = [x.split('/')[-1].split('.wav')[0]+'.png' for x in df['filename']]
    df['subset'] = ['train' if x in [1,2,3,4] else 'val' for x in df['fold']]
    
    #process each wav file
    wav_files = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
    for wav_file in wav_files:
        output_filename = define_image_filepath(wav_file, df, output_dir)
        create_spectrogram(wav_file, window, hop_length, time_steps, output_filename)
        
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',
                        required=False,
                        default='/data/datasets/esc50/ESC-50-master/audio/',
                        help='''Directory where .wav files live''')
    parser.add_argument('--window',
                        default=True,
                        action="store_true",
                        help='''True/false to have a window of the spectrogram''')
    parser.add_argument('--hop_length',
                        required=False,
                        default=512,
                        help='''Number of samples per time-step in spectogram''')
    parser.add_argument('--time_steps',
                        required=False,
                        default=200,
                        help='''Number of time-steps, width of image''')
    parser.add_argument('--output_dir',
                        required=False,
                        default='/data/datasets/esc50/esc50_processed/images/',
                        help='''Directory where to save the .png files''')

    args = parser.parse_args()

    main(input_dir=args.input_dir, window=args.window,
         hop_length=args.hop_length,time_steps=args.time_steps,
         output_dir=args.output_dir)
