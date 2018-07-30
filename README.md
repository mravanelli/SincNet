Work in Progress. The final version of the code will be available in few days.

# SincNet
SincNet is a neural architecture for processing **raw audio samples**. It is a novel Convolutional Neural Network (CNN) that encourages the first convolutional layer to discover more **meaningful filters**. SincNet is based on parametrized sinc functions, which implement band-pass filters.

In contrast to standard CNNs, that learn all elements of each filter, only low and high cutoff frequencies are directly learned from data with the proposed method. This offers a very compact and efficient way to derive a **customized filter bank** specifically tuned for the desired application. 

This project releases a collection of codes and utilities to perform speaker identification with SincNet.
An example of speaker identification with the TIMIT database is provided. 

<img src="https://github.com/mravanelli/SincNet/blob/master/SincNet.png" width="400" img align="right">

## Cite us
If you use this code or part of it, please cite us!

*Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet” Arxiv*


## Prerequisites
- Linux
- Python 3.6/2.7
- pytorch 0.4.0
- pysoundfile (``` conda install -c conda-forge pysoundfile```)
- We also suggest to use the anaconda environment.


## How to run a TIMIT experiment
Even though the code can be easily adapted to any speech dataset, in the following part of the documentation we provide an example based on the popular TIMIT dataset.

**1. Run TIMIT data preparation.**

This step is necessary to store a version of TIMIT in which start and end silences are removed and the amplitute of each speech utterance is normalized. To do it, run the following code:

``
python TIMIT_preparation.py $TIMIT_FOLDER $OUTPUT_FOLDER data_lists/TIMIT_all.scp
``

where:
- *$TIMIT_FOLDER* is the folder of the original TIMIT corpus
- *$OUTPUT_FOLDER* is the folder in which the normalized TIMIT will be stored
- *data_lists/TIMIT_all.scp* is the list of the TIMIT files used for training/test the speaker id system.

**2. Run the speaker id experiment.**

- Modify the *[data]* section of *cfg/SincNet_TIMIT.cfg* file according to your paths. In particular, modify the *data_folder* with the *$OUTPUT_FOLDER* specified during the TIMIT preparation. The other parameters of the config file belong to the following sections:
 1. *[windowing]*, that defines how each sentence is splitted into smaller chunks.
 2. *[cnn]*,  that specifies the characteristics of the CNN architecture.
 3. *[dnn]*,  that specifies the characteristics of the fully-connected DNN architecture following the CNN layers.
 4. *[class]*, that specify the softmax classification part.
 5. *[optimization]*, that reports the main hyperparameters used to train the architecture.

- Once setup the cfg file, you can run the speaker id experiments using the following command:

``
python speaker_id.py --cfg=cfg/SincNet_TIMIT.cfg
``

The network might take several hours to converge (depending on the speed of your GPU card). In our case, using an *nvidia TITAN X*, the full training took about 24 hours. If you use the code within a cluster is crucial to copy the normalized dataset into the local node, since the current version of the code requires frequent accesses to the stored wav files. Note that several possible optimizations to improve the code speed are not implemented in this version, since are out of the scope of this work.

**3. Results.**

The results are saved into the *output_folder* specified in the cfg file. In this folder, you can find a file (*res.res*) summarizing training and test error rates. The model *model_raw.pkl* is the SincNet model saved after the last iteration. 
Using the cfg file specified above, we obtain the following results:
```
epoch 0, loss_tr=6.309807 err_tr=0.997656 loss_te=6.332657 err_te=0.997605 err_te_snt=0.996392
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
epoch 80, loss_tr=0.111940 err_tr=0.033389 loss_te=2.925527 err_te=0.428739 err_te_snt=0.011544
epoch 88, loss_tr=0.101788 err_tr=0.031016 loss_te=3.050507 err_te=0.438099 err_te_snt=0.011544
epoch 96, loss_tr=0.089672 err_tr=0.027451 loss_te=3.212288 err_te=0.445679 err_te_snt=0.011544
epoch 104, loss_tr=0.085366 err_tr=0.026445 loss_te=3.226385 err_te=0.431996 err_te_snt=0.012266
epoch 112, loss_tr=0.077404 err_tr=0.023564 loss_te=3.341498 err_te=0.433145 err_te_snt=0.010101
epoch 120, loss_tr=0.073497 err_tr=0.022861 loss_te=3.858381 err_te=0.472951 err_te_snt=0.028139
epoch 128, loss_tr=0.067383 err_tr=0.020527 loss_te=3.474988 err_te=0.431545 err_te_snt=0.008658
epoch 136, loss_tr=0.064087 err_tr=0.019961 loss_te=3.341287 err_te=0.436171 err_te_snt=0.007215
epoch 144, loss_tr=0.062003 err_tr=0.019160 loss_te=3.412609 err_te=0.426363 err_te_snt=0.009380
epoch 152, loss_tr=0.058740 err_tr=0.018281 loss_te=3.815553 err_te=0.443672 err_te_snt=0.010823
epoch 160, loss_tr=0.055162 err_tr=0.017314 loss_te=3.784261 err_te=0.446239 err_te_snt=0.008658
epoch 168, loss_tr=0.053430 err_tr=0.016279 loss_te=3.397493 err_te=0.427959 err_te_snt=0.009380
epoch 176, loss_tr=0.052093 err_tr=0.016064 loss_te=3.777609 err_te=0.442838 err_te_snt=0.011544
epoch 184, loss_tr=0.050022 err_tr=0.015605 loss_te=3.615857 err_te=0.431436 err_te_snt=0.009380
epoch 192, loss_tr=0.048606 err_tr=0.014844 loss_te=4.254653 err_te=0.458577 err_te_snt=0.020924
epoch 200, loss_tr=0.045252 err_tr=0.014209 loss_te=3.809854 err_te=0.437975 err_te_snt=0.010101
epoch 208, loss_tr=0.046115 err_tr=0.014219 loss_te=3.525989 err_te=0.416244 err_te_snt=0.010823
epoch 216, loss_tr=0.046525 err_tr=0.013945 loss_te=3.731409 err_te=0.428357 err_te_snt=0.010101
epoch 224, loss_tr=0.043378 err_tr=0.013350 loss_te=4.014791 err_te=0.430589 err_te_snt=0.013709
epoch 232, loss_tr=0.042941 err_tr=0.013203 loss_te=3.774163 err_te=0.415966 err_te_snt=0.010101
epoch 240, loss_tr=0.040990 err_tr=0.012598 loss_te=3.788815 err_te=0.416591 err_te_snt=0.010823
epoch 248, loss_tr=0.039575 err_tr=0.011924 loss_te=3.918533 err_te=0.427865 err_te_snt=0.008658
epoch 256, loss_tr=0.038113 err_tr=0.011924 loss_te=3.933329 err_te=0.432080 err_te_snt=0.008658
epoch 264, loss_tr=0.038549 err_tr=0.011914 loss_te=3.887040 err_te=0.416849 err_te_snt=0.010823
epoch 272, loss_tr=0.039867 err_tr=0.012109 loss_te=4.017699 err_te=0.430378 err_te_snt=0.008658
epoch 280, loss_tr=0.037822 err_tr=0.011914 loss_te=4.395680 err_te=0.453985 err_te_snt=0.014430
epoch 288, loss_tr=0.036721 err_tr=0.011250 loss_te=4.222330 err_te=0.442820 err_te_snt=0.010101
epoch 296, loss_tr=0.035290 err_tr=0.010723 loss_te=3.918045 err_te=0.410693 err_te_snt=0.007937
epoch 304, loss_tr=0.034258 err_tr=0.010225 loss_te=4.165709 err_te=0.434250 err_te_snt=0.007215
epoch 312, loss_tr=0.034672 err_tr=0.010830 loss_te=4.313679 err_te=0.445955 err_te_snt=0.014430
epoch 320, loss_tr=0.033052 err_tr=0.009639 loss_te=4.076542 err_te=0.416710 err_te_snt=0.006494
epoch 328, loss_tr=0.033344 err_tr=0.010117 loss_te=3.928874 err_te=0.415024 err_te_snt=0.007215
epoch 336, loss_tr=0.033228 err_tr=0.010166 loss_te=4.030224 err_te=0.410034 err_te_snt=0.005051
epoch 344, loss_tr=0.033313 err_tr=0.010166 loss_te=4.402949 err_te=0.428691 err_te_snt=0.009380
epoch 352, loss_tr=0.031828 err_tr=0.009238 loss_te=4.080747 err_te=0.414066 err_te_snt=0.006494
epoch 360, loss_tr=0.033095 err_tr=0.009600 loss_te=4.254683 err_te=0.419954 err_te_snt=0.005772
``` 

## Where SincNet is implemented?
To take a look into the SincNet implementation you should open the file *dnn_models.py* and read the classes *SincNet*, *sinc_conv* and the function *sinc*.

## References

[1] Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet” Arxiv
