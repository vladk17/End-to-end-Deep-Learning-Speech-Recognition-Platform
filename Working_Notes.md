# Working Notes and the Plan

## Notes:
1. [Getting-Started by NVIDIA](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition#getting-started)
2. What is kaldi and what should we know about it?
3. https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition,<br> 
https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/jasper.html, <br>
https://arxiv.org/pdf/1904.03288.pdf, <br>
https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper
4. https://github.com/facebookresearch/wav2letter (try also to get info from FB wav2letter Users group https://www.facebook.com/groups/717232008481207/)
5. https://news.developer.nvidia.com/new-asr-model-speech-toolkit-interspeech2019/ ,<br>https://github.com/NVIDIA/NeMo, https://devblogs.nvidia.com/neural-modules-for-speech-language-models/,<br> 
["Jasper: An End-to-End Convolutional Neural Acoustic Model"](https://arxiv.org/pdf/1904.03288.pdf),<br> https://github.com/NVIDIA/NeMo/blob/master/examples/asr/notebooks/1_ASR_tutorial_using_NeMo.ipynb
6. [End2End Models](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition):
* DeepSpeech2
* Wave2Letter+ (wav2letter installation is not trivial)
* Jasper DR 10x5
7. Datasets:
* 7.1. [LibriSpeech ASR corpus](http://www.openslr.org/12) is a corpus of approximately 1000 hours of 16kHz read English speech. 
The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned.<br>
Acoustic models, trained on this data set, are available at kaldi-asr.org and language models, suitable for evaluation can be found at http://www.openslr.org/11/.

    The paper ["LibriSpeech: an ASR corpus based on public domain audio books"](http://www.danielpovey.com/files/2015_icassp_librispeech.pdf)

* 7.2 Wall Street Journal and the Hub5’00 conversational evaluation datasets

8. [wav2vec: Unsupervised Pre-training for Speech Recognition](https://research.fb.com/publications/wav2vec-unsupervised-pre-training-for-speech-recognition/)
9. [Wav2vec: State-of-the-art speech recognition through self-supervision](https://ai.facebook.com/blog/wav2vec-state-of-the-art-speech-recognition-through-self-supervision/)
10. [Self-supervision and building more robust speech recognition systems](https://ai.facebook.com/blog/self-supervision-and-building-more-robust-speech-recognition-systems/)
11. [Deep Learning for Speech Recognition (Adam Coates, Baidu)](https://www.youtube.com/watch?v=g-sndkf7mCs&t=937s)
12. [Mixed Precision Training for NLP and Speech Recognition with OpenSeq2Seq](https://devblogs.nvidia.com/mixed-precision-nlp-speech-openseq2seq/?fbclid=IwAR3liPZgoBM5lboHFiA4uNxE6YWOCblFal-odajiBN5SdMOAz7eIhWFHHLM)
13. [Develop Smaller Speech Recognition Models with NVIDIA’s NeMo Framework](https://devblogs.nvidia.com/develop-smaller-speech-recognition-models-with-nvidias-nemo-framework/)
14. [BERT](https://github.com/google-research/bert)
15. [Improving speech recognition with BERTx2 post-processing model (NeMo)](https://nvidia.github.io/NeMo/nlp/asr-improvement.html)
16. [Building Spanish N-Gram Language Model with KenLM](https://yidatao.github.io/2017-05-31/kenlm-ngram/)


## AWS:
1. [AWS Deep Learning AMI](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html)

## Plan:
1. Set up training environment in the AWS cloud 
* 1.1. Questions: type of machine, cluster, type and number of GPUs... (small and large configurations)
* 1.2. Reproduce "example notebook" content on AWS machine (in notebook and plane python script)
2. Know the elements of the Pipeline/Model from the source code of NeMo, DeepSpeech decoders, kenlm.  
3. Jasper and QuartzNet - know their architectures (from the source code). Know how to change configurations. Know how to work with pretrained models. Know how to work with greedy CTC and with language models. In colab and AWS.  
4. Transfer Learning [link](https://devblogs.nvidia.com/how-to-build-domain-specific-automatic-speech-recognition-models-on-gpus/)
5. Get numbers: measure WER, learning/inference time, etc.

