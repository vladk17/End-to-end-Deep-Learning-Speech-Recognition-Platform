# Working Notes and the Plan

## Notes:
1. [LibriSpeech ASR corpus](http://www.openslr.org/12) is a corpus of approximately 1000 hours of 16kHz read English speech. 
The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned.<br>
Acoustic models, trained on this data set, are available at kaldi-asr.org and language models, suitable for evaluation can be found at http://www.openslr.org/11/.
2. What is kaldi and what should we know about it?
3. https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/jasper.html, https://arxiv.org/pdf/1904.03288.pdf, https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper
4. https://github.com/facebookresearch/wav2letter 


## Plan:
1. Need a baseline model and pipeline
2. Start with wav2letter as it looks to be more user friendly at the first glance
