Code for DrivenData [Literacy Screening comptetition](https://www.drivendata.org/competitions/298/literacy-screening/leaderboard/) - gets 10th place on private LB (it aint much but its honest work).

Approach is the following:
* Extract phonemes from audio.
* Extract phonemes from expected text.
* Transcript audio using Whisper.
* Calcualte a bunch of features based on extracted data (e.g. levenstein distance between extracted and expected phonemes, same for texts, etc.).
* Extract features from audio.
* Generate features based on metadata (e.g. frequency and success rate for given texts).
* Put everything in XGBoost.

I had no time to try training something e2e or encode audio and expected text together, etc.; so I just stopped on generate lots of stuff and let GBM handle approach. This works, but it is definetly behind leading solutions.

