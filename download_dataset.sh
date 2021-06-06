cd data/
wget https://zenodo.org/record/4743815/files/clotho_audio_development.7z
wget https://zenodo.org/record/4743815/files/clotho_audio_evaluation.7z
wget https://zenodo.org/record/4743815/files/clotho_audio_validation.7z
7z x clotho_audio_development.7z
7z x clotho_audio_validation.7z
7z x clotho_audio_evaluation.7z
rm -rf clotho_audio_validation.7z
rm -rf clotho_audio_development.7z
rm -rf clotho_audio_evaluation.7z
python mv_wavs.py
