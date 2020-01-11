# End-to-end-Deep-Learning-Speech-Recognition-Platform


## Problem Description:
Automatic Speech Recognition (ASR, aka Speech-to-Text), is historically comprised
of two parts – an Acoustic Model, mapping speech to phonemes, and a Language
Model, mapping phonemes to words. While in the fields of Computer Vision and
NLP, deep learning models outperform other approaches by large margins, in the
field of ASR the vast majority of commercial systems don’t use deep learning.
In recent years, there have been reports on success in applying end-to-end
(e2e) deep learning architectures to transcribe speech. E2E DL platforms hold
the promise of better adaptation to accents or new languages, making the use
of phonetic lexicons obsolete. Two promising directions are __Nvidia’s Jasper__ and
__Facebook’s Wav2Letter__ architectures.

## Dataset Description:
Gong will provide a training set of 500 hours of English sales calls, manually
transcribed, and manually transcribed test sets of 10 hours in each language. A
publicly available dataset of Spanish will also be provided.

## Project Goals:
* Basic goal: Develop a Deep Learning based system for Automatic Speech
Recognition in Spanish. Given an audio wav file, the system should provide
its transcription in a factor of approx. x5 real time, along with time stamps for
each transcribed word.
* Advanced goal: Given a test set of 10+ hours of sales calls from the Gong data
in each language, reach Word Error Rate (WER) that is lower than that of the
Amazon Comprehend ASR system in the same language.

## Project Impact:
The project will allow Gong to improve its service in Spanish and English. The
quality of transcription affects the quality of downstream tasks as well, including
NLP understanding of the conversations. Gong’s existing Speech Recognition
system does not use Deep Learning, and reaches a very low WER compared with
commercial solution like Google or Amazon, and is key to Gong’s success in Natural
Language Understanding. Having a Deep Learning based network for ASR in some
languages will allow Gong to more easily support additional languages by training
them on relevant data.
