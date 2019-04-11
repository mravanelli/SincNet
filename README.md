
# SincNet
SincNet is a neural architecture for processing **raw audio samples**. It is a novel Convolutional Neural Network (CNN) that encourages the first convolutional layer to discover more **meaningful filters**. SincNet is based on parametrized sinc functions, which implement band-pass filters.

In contrast to standard CNNs, that learn all elements of each filter, only low and high cutoff frequencies are directly learned from data with the proposed method. This offers a very compact and efficient way to derive a **customized filter bank** specifically tuned for the desired application. 

This project releases a collection of codes and utilities to perform speaker identification with SincNet.
An example of speaker identification with the TIMIT database is provided. If you are interested in **SincNet applied to speech recognition you can take a look into the PyTorch-Kaldi github repository (https://github.com/mravanelli/pytorch-kaldi).** 

<img src="https://github.com/mravanelli/SincNet/blob/master/SincNet.png" width="400" img align="right">

[Take a look into our video introduction to SincNet](https://www.youtube.com/watch?v=mXQBObRGUgk&feature=youtu.be)

## Cite us
If you use this code or part of it, please cite us!

*Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet”* [Arxiv](http://arxiv.org/abs/1808.00158)


## Prerequisites
- Linux
- Python 3.6/2.7
- pytorch 1.0
- pysoundfile (``` conda install -c conda-forge pysoundfile```)
- We also suggest using the anaconda environment.

## Updates
Feb, 16 2019:
- We replaced the old "sinc_conv"  with "SincConv_fast". The latter is 50% faster.
- In the near future, we plan to support SincNet based speaker-id within the [PyTorch-Kaldi project](https://github.com/mravanelli/pytorch-kaldi) (the current version of the project only supports SincNEt for speech recognition experiments). This will allow users to perform speaker recognition experiments in a faster and much more flexible environment. The current repository will anyway remain as a showcase. 

## How to run a TIMIT experiment
Even though the code can be easily adapted to any speech dataset, in the following part of the documentation we provide an example based on the popular TIMIT dataset.

**1. Run TIMIT data preparation.**

This step is necessary to store a version of TIMIT in which start and end silences are removed and the amplitude of each speech utterance is normalized. To do it, run the following code:

``
python TIMIT_preparation.py $TIMIT_FOLDER $OUTPUT_FOLDER data_lists/TIMIT_all.scp
``

where:
- *$TIMIT_FOLDER* is the folder of the original TIMIT corpus
- *$OUTPUT_FOLDER* is the folder in which the normalized TIMIT will be stored
- *data_lists/TIMIT_all.scp* is the list of the TIMIT files used for training/test the speaker id system.

**2. Run the speaker id experiment.**

- Modify the *[data]* section of *cfg/SincNet_TIMIT.cfg* file according to your paths. In particular, modify the *data_folder* with the *$OUTPUT_FOLDER* specified during the TIMIT preparation. The other parameters of the config file belong to the following sections:
 1. *[windowing]*, that defines how each sentence is split into smaller chunks.
 2. *[cnn]*,  that specifies the characteristics of the CNN architecture.
 3. *[dnn]*,  that specifies the characteristics of the fully-connected DNN architecture following the CNN layers.
 4. *[class]*, that specify the softmax classification part.
 5. *[optimization]*, that reports the main hyperparameters used to train the architecture.

- Once setup the cfg file, you can run the speaker id experiments using the following command:

``
python speaker_id.py --cfg=cfg/SincNet_TIMIT.cfg
``

The network might take several hours to converge (depending on the speed of your GPU card). In our case, using an *nvidia TITAN X*, the full training took about 24 hours. If you use the code within a cluster is crucial to copy the normalized dataset into the local node, since the current version of the code requires frequent accesses to the stored wav files. Note that several possible optimizations to improve the code speed are not implemented in this version since are out of the scope of this work.


**3. Results.**

The results are saved into the *output_folder* specified in the cfg file. In this folder, you can find a file (*res.res*) summarizing training and test error rates. The model *model_raw.pkl* is the SincNet model saved after the last iteration. 
Using the cfg file specified above, we obtain the following results:
```
epoch 0, loss_tr=5.542032 err_tr=0.984189 loss_te=4.996982 err_te=0.969038 err_te_snt=0.919913
epoch 8, loss_tr=1.693487 err_tr=0.434424 loss_te=2.735717 err_te=0.612260 err_te_snt=0.069264
epoch 16, loss_tr=0.861834 err_tr=0.229424 loss_te=2.465258 err_te=0.520276 err_te_snt=0.038240
epoch 24, loss_tr=0.528619 err_tr=0.144375 loss_te=2.948707 err_te=0.534053 err_te_snt=0.062049
epoch 32, loss_tr=0.362914 err_tr=0.100518 loss_te=2.530276 err_te=0.469060 err_te_snt=0.015152
epoch 40, loss_tr=0.267921 err_tr=0.076445 loss_te=2.761606 err_te=0.464799 err_te_snt=0.023088
epoch 48, loss_tr=0.215479 err_tr=0.061406 loss_te=2.737486 err_te=0.453493 err_te_snt=0.010823
epoch 56, loss_tr=0.173690 err_tr=0.050732 loss_te=2.812427 err_te=0.443322 err_te_snt=0.011544
epoch 64, loss_tr=0.145256 err_tr=0.043594 loss_te=2.917569 err_te=0.438507 err_te_snt=0.009380
epoch 72, loss_tr=0.128894 err_tr=0.038486 loss_te=3.009008 err_te=0.438005 err_te_snt=0.019481
....
epoch 320, loss_tr=0.033052 err_tr=0.009639 loss_te=4.076542 err_te=0.416710 err_te_snt=0.006494
epoch 328, loss_tr=0.033344 err_tr=0.010117 loss_te=3.928874 err_te=0.415024 err_te_snt=0.007215
epoch 336, loss_tr=0.033228 err_tr=0.010166 loss_te=4.030224 err_te=0.410034 err_te_snt=0.005051
epoch 344, loss_tr=0.033313 err_tr=0.010166 loss_te=4.402949 err_te=0.428691 err_te_snt=0.009380
epoch 352, loss_tr=0.031828 err_tr=0.009238 loss_te=4.080747 err_te=0.414066 err_te_snt=0.006494
epoch 360, loss_tr=0.033095 err_tr=0.009600 loss_te=4.254683 err_te=0.419954 err_te_snt=0.005772
``` 
The converge is initially very fast (see the first 30 epochs). After that the performance improvement decreases and oscillations into the sentence error rate performance appear. Despite these oscillations an average improvement trend can be observed for the subsequent epochs. In this experiment, we stopped our training  at epoch 360.
The fields of the res.res file have the following meaning:
- loss_tr: is the average training loss (i.e., cross-entropy function) computed at every frame.
- err_tr: is the classification error (measured at frame level) of the training data. Note that we split the speech signals into chunks of 200ms with 10ms overlap. The error is averaged for all the chunks of the training dataset.
- loss_te is the average test loss (i.e., cross-entropy function) computed at every frame.
- err_te: is the classification error (measured at frame level) of the test data.
- err_te_snt: is the classification error (measured at sentence level) of the test data. Note that we split the speech signal into chunks of 200ms with 10ms overlap. For each chunk, our SincNet performs a prediction over the set of speakers. To compute this classification error rate we averaged the predictions and, for each sentence, we voted for the speaker with the highest average probability.

[You can find our trained model for TIMIT here.](https://bitbucket.org/mravanelli/sincnet_models/)

## Where SincNet is implemented?
To take a look into the SincNet implementation you should open the file *dnn_models.py* and read the classes *SincNet*, *sinc_conv* and the function *sinc*.

## How to use SincNet with a different dataset?
In this repository, we used the TIMIT dataset as a tutorial to show how SincNet works. 
With the current version of the code, you can easily use a different corpus. To do it you should provide in input the corpora-specific input files (in wav format) and your own labels. You should thus modify the paths into the *.scp files you find in the data_lists folder. 

To assign to each sentence the right label, you also have to modify the dictionary "*TIMIT_labels.npy*". 
The labels are specified within a python dictionary that contains sentence ids as keys (e.g., "*si1027*") and speaker_ids as values. Each speaker_id is an integer, ranging from 0 to N_spks-1. In the TIMIT dataset, you can easily retrieve the speaker id from the path (e.g., *train/dr1/fcjf0/si1027.wav* is the sentence_id "*si1027*" uttered by the speaker "*fcjf0*"). For other datasets, you should be able to retrieve in such a way this dictionary containing pairs of speakers and sentence ids.

You should then modify the config file (*cfg/SincNet_TIMIT.cfg*) according to your new paths. Remember also to change the field "*class_lay=462*" according to the number of speakers N_spks you have in your dataset.

**The version of the Librispeech dataset used in the paper is available upon request**. In our work, we have used only 12-15 seconds of training material for each speaker and we processed the original librispeech sentences in order to perform amplitude normalization. Moreover, we used a simple energy-based VAD to avoid silences at the beginning and end of each sentence as well as to split in multiple chunks the sentences that contain longer silence



## References

[1] Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet” [Arxiv](http://arxiv.org/abs/1808.00158)
