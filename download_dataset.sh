cd data/
wget https://zenodo.org/record/4783391/files/clotho_audio_development.7z
wget https://zenodo.org/record/4783391/files/clotho_audio_evaluation.7z
wget https://zenodo.org/record/4783391/files/clotho_audio_validation.7z
wget https://zenodo.org/record/4783391/files/clotho_captions_development.csv
wget https://zenodo.org/record/4783391/files/clotho_captions_evaluation.csv
wget https://zenodo.org/record/4783391/files/clotho_captions_validation.csv
mv clotho_captions_* csv_files/
7z x clotho_audio_development.7z
7z x clotho_audio_validation.7z
7z x clotho_audio_evaluation.7z
rm -rf clotho_audio_validation.7z
rm -rf clotho_audio_development.7z
rm -rf clotho_audio_evaluation.7z

