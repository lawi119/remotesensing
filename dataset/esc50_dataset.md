## ESC 50 dataset

### Original dataset link
* https://github.com/karolpiczak/ESC-50

### Azure commercial location
* Subscription: MSCE-AI-TEAM-COMMERCIAL
* Resource group: esc_50
* Storage account: ecs50data
* Container: data

### Raw data
* ESC-50-master/audio: 2000 .wav files
* ESC-50-master/meta/esc50.csv: labels for 2000 .wav files

### Processed data for model
* esc50_processed2/images: 2000 .png files created using preprocess/convert_wav_to_spectogram.py
