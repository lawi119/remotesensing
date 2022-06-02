## ESC 50 dataset
### Azure Commercial
* Subscription: MSCE_AI_TEAM_COMMERCIAL
* Resource group: esc_50
* Storage account: ecs50data
* Container: data

### Raw data
* ESC-50-master/audio: 2000 .wav files
* ESC-50-master/meta/esc50.csv: labels for 2000 .wav files

### Processed data for model
* esc50_processed/images: 2000 .png files created using preprocess/convert_wav_to_spectogram.py
* esc50_processed/meta/esc50_train.csv: train labels for 1600 .wav files created using preprocess/split_meta.py
* esc50_processed/meta/esc50_test.csv: test labels for 400 .wav files created using preprocess/split_meta.py