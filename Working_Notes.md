# Working Notes and the Plan

## Notes:
1. [OpenSeq2Seq by NVIDIA Getting-Started](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition#getting-started)
2. https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition,<br> 
https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/jasper.html, <br>
Scripts for Jasper model to achieve near state of the art accuracy and perform high-performance inference using NVIDIA TensorRT: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper
3. Kaldi: http://kaldi-asr.org/, https://github.com/kaldi-asr/kaldi; [GPU-Accelerated Speech to Text with Kaldi: A Tutorial on Getting Started](https://devblogs.nvidia.com/gpu-accelerated-speech-to-text-with-kaldi-a-tutorial-on-getting-started/) by NVIDIA; 
[PyTorch-Kaldi](https://github.com/mravanelli/pytorch-kaldi) - an open-source repository for developing state-of-the-art DNN/HMM speech recognition systems. The DNN part is managed by PyTorch, while feature extraction, label computation, and decoding are performed with the Kaldi toolkit.
4. https://github.com/facebookresearch/wav2letter (try also to get info from FB wav2letter Users group https://www.facebook.com/groups/717232008481207/)
5. https://news.developer.nvidia.com/new-asr-model-speech-toolkit-interspeech2019/ ,<br>https://github.com/NVIDIA/NeMo, https://devblogs.nvidia.com/neural-modules-for-speech-language-models/,<br> 
__Jasper__ (Just Another Speech Recognizer) is a deep time delay neural network ([TDNN](https://en.wikipedia.org/wiki/Time_delay_neural_network)) comprising of blocks of 1D-convolutional layers <br> ["Jasper: An End-to-End Convolutional Neural Acoustic Model"](https://arxiv.org/pdf/1904.03288.pdf), Li et al,<br> https://github.com/NVIDIA/NeMo/blob/master/examples/asr/notebooks/1_ASR_tutorial_using_NeMo.ipynb
6. Frameworks:
* [NeMo](https://github.com/NVIDIA/NeMo)
* [OpenSeq2Seq](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition)
* [DeepSpeech2](https://github.com/PaddlePaddle/DeepSpeech)
* [Wave2Letter+](https://github.com/facebookresearch/wav2letter) (wav2letter installation is not trivial); [introductory presentation](https://www.infoq.com/presentations/wav2letter-facebook/); ["Wav2Letter: an End-to-End ConvNet-based Speech Recognition System"](https://arxiv.org/pdf/1609.03193.pdf), Collobert et. al; ["Letter-Based Speech Recognition with Gated ConvNets"](https://arxiv.org/pdf/1712.09444.pdf), Liptchinsky et. al ;see also [pytorch implementation](https://www.facebook.com/groups/461707137729175/permalink/639744036592150/)
* [ESPnet](https://github.com/espnet/espnet); ["ESPnet: End-to-End Speech Processing Toolkit"](https://arxiv.org/pdf/1804.00015.pdf), Watanabe et. al; see also ["Very Deep Convolutional Networks for End-To-End Speech Recognition"](https://arxiv.org/pdf/1610.03022.pdf), Zhang et. al.; ["Advances in Joint CTC-Attention based End-to-End Speech Recognition with
a Deep CNN Encoder and RNN-LM"](https://arxiv.org/pdf/1706.02737.pdf), Hori et al.; [ESPnet: interspeech2019-tutorial (github repo)](https://github.com/espnet/interspeech2019-tutorial)<br>
    * ESPnet ASR implementation is based on transformer:<br>
    The Annotated Transformer: [paper](http://nlp.seas.harvard.edu/2018/04/03/attention.html), [github](https://github.com/harvardnlp/annotated-transformer), [colab](https://drive.google.com/file/d/1xQXSv6mtAOLXxEMi8RvaW8TW-7bvYBDF/view)
* [SpeechBrain] (https://speechbrain.github.io/)(by Mila) - so far no publicly shared code, just declarations; [Machine Translation Group at The Johns Hopkins University](http://www.statmt.org/jhu/?n=Main.HomePage)
7. Datasets: <br>
> [Audio Data Links](https://github.com/robmsmt/ASR_Audio_Data_Links), a list of publically and privately available audio data for ASR
* 7.1. [LibriSpeech ASR corpus](http://www.openslr.org/12) is a corpus of approximately 1000 hours of 16kHz read English speech. 
The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned.<br>
Acoustic models, trained on this data set, are available at kaldi-asr.org and language models, suitable for evaluation can be found at http://www.openslr.org/11/.

    The paper ["LibriSpeech: an ASR corpus based on public domain audio books"](http://www.danielpovey.com/files/2015_icassp_librispeech.pdf)

* 7.2 Wall Street Journal and the Hub5’00 conversational evaluation datasets

WSJ is part of the Linguistic Data Consortium and can be found here:<br>
CSR-I (WSJ0) Complete: https://catalog.ldc.upenn.edu/LDC93S6A<br>
CSR-II (WSJ1) Complete: https://catalog.ldc.upenn.edu/LDC94S13A<br>
Note: To download this dataset a license is required please refer to [LDC to learn more](https://www.ldc.upenn.edu/language-resources/data/obtaining).<br>
Wav2Letter recipes for WSJ is [here](https://github.com/facebookresearch/wav2letter/tree/master/recipes/data/wsj) 



8. [wav2vec: Unsupervised Pre-training for Speech Recognition](https://research.fb.com/publications/wav2vec-unsupervised-pre-training-for-speech-recognition/)
9. [Wav2vec: State-of-the-art speech recognition through self-supervision](https://ai.facebook.com/blog/wav2vec-state-of-the-art-speech-recognition-through-self-supervision/)
10. [Self-supervision and building more robust speech recognition systems](https://ai.facebook.com/blog/self-supervision-and-building-more-robust-speech-recognition-systems/)
11. [Deep Learning for Speech Recognition (Adam Coates, Baidu)](https://www.youtube.com/watch?v=g-sndkf7mCs&t=937s)
12. [Mixed Precision Training for NLP and Speech Recognition with OpenSeq2Seq](https://devblogs.nvidia.com/mixed-precision-nlp-speech-openseq2seq/?fbclid=IwAR3liPZgoBM5lboHFiA4uNxE6YWOCblFal-odajiBN5SdMOAz7eIhWFHHLM)
13. [Develop Smaller Speech Recognition Models with NVIDIA’s NeMo Framework](https://devblogs.nvidia.com/develop-smaller-speech-recognition-models-with-nvidias-nemo-framework/)
14. [BERT](https://github.com/google-research/bert)
15. [Improving speech recognition with BERTx2 post-processing model (NeMo)](https://nvidia.github.io/NeMo/nlp/asr-improvement.html)
16. [Building Spanish N-Gram Language Model with KenLM](https://yidatao.github.io/2017-05-31/kenlm-ngram/), [Understanding ARPA and Language Models](https://medium.com/@canadaduane/understanding-arpa-and-language-models-115d6cbc3893), [ARPA Language models](https://cmusphinx.github.io/wiki/arpaformat/)
17. [Domain Specific – NeMo ASR Application. NVIDIA docker](https://ngc.nvidia.com/catalog/containers/nvidia:nemo_asr_app_img)
18. [NeMo Models collection](https://ngc.nvidia.com/catalog/models?orderBy=modifiedDESC&query=nemo&quickFilter=models&filters=)
19. [NeMo Containers collection](https://ngc.nvidia.com/catalog/containers?orderBy=modifiedDESC&pageNumber=0&query=nemo&quickFilter=containers&filters=)
20. [Patter](https://github.com/ryanleary/patter) speech-to-text framework in PyTorch with initial support for the DeepSpeech2 architecture (and variants of it).
21. [Connectionist Temporal Classification: Labelling Unsegmented
Sequence Data with Recurrent Neural Networks](http://www.cs.toronto.edu/~graves/icml_2006.pdf), Graves et al
22. ["A Fully Differentiable Beam Search Decoder"](https://arxiv.org/abs/1902.06022), Collobert et al. 
23. ["Multichannel End-to-end Speech Recognition"](https://arxiv.org/pdf/1703.04783.pdf), Ochiai et al.
24. ["Language independent end-to-end architecture for joint language identification and speech recognition"](https://www.merl.com/publications/docs/TR2017-182.pdf), Watanabe et al
25. [Discossion on "ASR for German (or any other language that is not English)"](https://github.com/NVIDIA/OpenSeq2Seq/issues/497)
26. [NVIDIA: Containers For Deep Learning Frameworks User Guide](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html)
27. [NVIDIA Deep Learning Frameworks](https://docs.nvidia.com/deeplearning/frameworks/index.html)
28. [NVIDIA GPU Cloud on AWS setup guide](https://docs.nvidia.com/ngc/ngc-aws-setup-guide/index.html)


### AWS:
1. [AWS Deep Learning AMI](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html)
2. [Launch an AWS Deep Learning AMI](https://aws.amazon.com/getting-started/tutorials/get-started-dlami/)<br>
installation commands for ec2 deep learning base ubuntu instance:
```
wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O "Miniconda3-latest-Linux-x86_64.sh"
bash "Miniconda3-latest-Linux-x86_64.sh" -b
echo "export PATH=\"$HOME/miniconda3/bin:\$PATH\"" >> ~/.bashrc
export PATH="$HOME/miniconda3/bin:$PATH"
vim environment.yml

name: py36

channels:
  - conda-forge
  - defaults
dependencies:
  - python = 3.6
  - jupyter
  - matplotlib = 2.0.2
  - pandas = 0.20.1
  - numpy = 1.12.1
  - scikit-learn = 0.18.1
  
conda env create -f environment.yml
conda init bash
conda activate py36
```
ssh with local port mapping for jupyter (from git cli, cygwin, etc.):
```
ssh -L localhost:8888:localhost:8888 -i <your .pem filename> ubuntu@<your instance DNS>
```

3. [EC2 User Guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html)
4. [AWS Command Line Interface](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html)
5. [PYTORCH 1.0 DISTRIBUTED TRAINER WITH AMAZON AWS](https://pytorch.org/tutorials/beginner/aws_distributed_training_tutorial.html)<br>
    5.1 [WRITING DISTRIBUTED APPLICATIONS WITH PYTORCH](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
6. [AWS Pricing](https://aws.amazon.com/ec2/pricing/)

## Questions and TBDs:
1. Find a way to support time-stamping of transcribed words
2. Profiling of computation time for learning and inference

## Plan:
1. Set up training environment in the AWS cloud 
  * 1.1. Questions: type of machine, cluster, type and number of GPUs... (small and large configurations)
  * 1.2. Reproduce "example notebook" content on AWS machine (in notebook and plane python script)
2. Know the elements of the Pipeline/Model from the source code of NeMo, DeepSpeech decoders, kenlm.
  * 2.1. Study NeMo scripts as examples of good practice 
3. Jasper and QuartzNet - know their architectures (from the source code). Know how to change configurations. Know how to work with pretrained models. Know how to work with greedy CTC and with language models. In colab and AWS.  
4. Transfer Learning [link](https://devblogs.nvidia.com/how-to-build-domain-specific-automatic-speech-recognition-models-on-gpus/)
Recomendatins:
  * 4.1 Take a model that was pre-trained on LibriSpeech, and use it to predict the transcript of a different dataset online (e.g. AN4 or a different set)
      * 4.1.1. Make sure you can fine-tune a trained model. Start with the model NVidia already trained on LibriSpeech - https://nvidia.github.io/NeMo/asr/tutorial.html#fine-tuning 
      * 4.1.2. Use code to evaluate performance, including checking the specific differences in predictions - we have code for that here -  https://github.com/gong-io/wer_analysis and I'll share more info about it later
      * 4.1.3. Add a Domain-specific Language Model, using KenLM (there are guides for it in the NeMo documentation)<br>
      example:
      ```
      * cd ``<nemo_git_root>/scripts``
      * `./install_decoders.sh`
      * `./build_6-gram_OpenSLR_lm.sh`
      ```
      * 4.1.4. Use n-gram and transformer based decoders instead of a greedy decoder

#### First Milestone: "Run a pretrained DL network to make predictions on LibriSpeech.Fine tune LM for WSJ dataset"

The milestone Based on [Domain Specific – NeMo ASR Application](https://ngc.nvidia.com/catalog/containers/nvidia:nemo_asr_app_img) from NVIDIA, which provides docker image that contains the complete Domain Specific NeMo ASR application (including notebooks, tools and scripts). <br>
quartznet15x5 model pretrained on LibriSpeech and Mozilla's EN Common Voice "validated" set is [here](https://ngc.nvidia.com/catalog/models/nvidia:quartznet15x5). We will tune this model for WSJ dataset.<br>
we will compare the preformance and the parameters of our tuned model with the model tuned by NVDIA, which is located [here](https://ngc.nvidia.com/catalog/models/nvidia:wsj_quartznet_15x5)
