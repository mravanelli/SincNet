#!/usr/bin/env python3

# TIMIT_preparation 
# Mirco Ravanelli 
# Mila - University of Montreal 

# July 2018

# Description: 
# This code prepares TIMIT for the following speaker identification experiments. 
# It removes start and end silences according to the information reported in the *.wrd files and normalizes the amplitude of each sentence.
 
# How to run it:
# python TIMIT_preparation.py $TIMIT_FOLDER $OUTPUT_FOLDER data_lists/TIMIT_all.scp 

# NOTE: This script expects filenames in lowercase (e.g, train/dr1/fcjf0/si1027.wav" rather than "TRAIN/DR1/FCJF0/SI1027.WAV)


import shutil
import os
import soundfile as sf
import numpy as np
import sys

def ReadList(list_file):
 f=open(list_file,"r")
 lines=f.readlines()
 list_sig=[]
 for x in lines:
    list_sig.append(x.rstrip())
 f.close()
 return list_sig

def copy_folder(in_folder,out_folder):
 if not(os.path.isdir(out_folder)):
  shutil.copytree(in_folder, out_folder, ignore=ig_f)

def ig_f(dir, files):
 return [f for f in files if os.path.isfile(os.path.join(dir, f))]



in_folder=sys.argv[1]
out_folder=sys.argv[2]
list_file=sys.argv[3]

# Read List file
list_sig=ReadList(list_file)

# Replicate input folder structure to output folder
copy_folder(in_folder,out_folder)


# Speech Data Reverberation Loop
for i in range(len(list_sig)): 
 
 # Open the wav file
 wav_file=in_folder+'/'+list_sig[i]
 [signal, fs] = sf.read(wav_file)
 signal=signal.astype(np.float64)

 # Signal normalization
 signal=signal/np.max(np.abs(signal))

 # Read wrd file
 wrd_file=wav_file.replace(".wav",".wrd")
 wrd_sig=ReadList(wrd_file)
 beg_sig=int(wrd_sig[0].split(' ')[0])
 end_sig=int(wrd_sig[-1].split(' ')[1])
 
 # Remove silences
 signal=signal[beg_sig:end_sig]

 
 # Save normalized speech
 file_out=out_folder+'/'+list_sig[i]

 sf.write(file_out, signal, fs)
 
 print("Done %s" % (file_out))
