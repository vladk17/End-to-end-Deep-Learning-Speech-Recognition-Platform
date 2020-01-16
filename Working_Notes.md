# Working Notes and the Plan

## Notes:
1. [LibriSpeech ASR corpus](http://www.openslr.org/12) is a corpus of approximately 1000 hours of 16kHz read English speech. 
The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned.<br>
Acoustic models, trained on this data set, are available at kaldi-asr.org and language models, suitable for evaluation can be found at http://www.openslr.org/11/.
2. What is kaldi and what should we know about it?
3. https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition,<br> https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/jasper.html, <br>https://arxiv.org/pdf/1904.03288.pdf, <br>https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper
4. https://github.com/facebookresearch/wav2letter (try also to get info from FB wav2letter Users group https://www.facebook.com/groups/717232008481207/)
5. https://news.developer.nvidia.com/new-asr-model-speech-toolkit-interspeech2019/ ,<br>https://github.com/NVIDIA/NeMo, https://devblogs.nvidia.com/neural-modules-for-speech-language-models/,<br> 
["Jasper: An End-to-End Convolutional Neural Acoustic Model"](https://arxiv.org/pdf/1904.03288.pdf),<br> https://github.com/NVIDIA/NeMo/blob/master/examples/asr/notebooks/1_ASR_tutorial_using_NeMo.ipynb
6. wav2letter installation is not trivial
7. Datasets:
* LibriSpeech
* Wall Street Journal and the Hub5’00 conversational evaluation datasets
8.

## Plan:
1. Set-up the repository and the data for the baseline model and pipline on local machine and on cloud vm (what cloud? what basic configuration to create?)
2. Define the E2E ASR pipeline block diagram
3. Define milestones
> 1. Discuss ["Jasper: An End-to-End Convolutional Neural Acoustic Model"](https://arxiv.org/pdf/1904.03288.pdf)
> * 1.1 Jasper Architecture - Select the simplest ("Our largest version uses 54 convolutional layers (333M parameters), while our smaller model uses 34 (201M parameters)")
> 2. Reproduce ["Jasper: An End-to-End Convolutional Neural Acoustic Model"](https://arxiv.org/pdf/1904.03288.pdf)
> 2.1 Setup Minimal configuration
> 2.3 Verify that the WER is not far from 3%

