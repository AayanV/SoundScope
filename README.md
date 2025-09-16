# SoundScope (Clean Restart)
SoundScope is a machine learning pipeline that predicts whether a Spotify track will be a “hit” based on its audio features.

The project connects to Spotify’s API to collect track data from playlists, searches, or new releases. That data is processed into features, then used to train regression and classification models. The models evaluate track popularity (using a threshold score like 75) and produce metrics such as accuracy, R², and feature importance plots.
