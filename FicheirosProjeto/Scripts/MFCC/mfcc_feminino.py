# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:56:33 2021

@author: nezfa
"""

import librosa
from praatio import tgio
import librosa.display
import matplotlib.pyplot as plt
from dtw import dtw
from numpy.linalg import norm
import numpy as np

tier_name = 'annotations'

#%% Ficheiros wav e TextGrid

tg1=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\F_free_ninho\gs_3.TextGrid")
tg2=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\F_nose_ninho\gc_3.TextGrid")
f1=r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\WAV\F_free_ninho\gs_3.wav"
f2=r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\WAV\F_nose_ninho\gc_3.wav"

tg3=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\F_free_ninho\gs_6.TextGrid")
tg4=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\F_nose_ninho\gc_6.TextGrid")
f3=r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\WAV\F_free_ninho\gs_6.wav"
f4=r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\WAV\F_nose_ninho\gc_6.wav"

tg5=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\F_free_ninho\gs_15.TextGrid")
tg6=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\F_nose_ninho\gc_15.TextGrid")
f5=r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\WAV\F_free_ninho\gs_15.wav"
f6=r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\WAV\F_nose_ninho\gc_15.wav"

tg7=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\F_free_ninho\gs_21.TextGrid")
tg8=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\F_nose_ninho\gc_21.TextGrid")
f7=r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\WAV\F_free_ninho\gs_21.wav"
f8=r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\WAV\F_nose_ninho\gc_21.wav"

tg9=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\F_free_ninho\gs_27.TextGrid")
tg10=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\F_nose_ninho\gc_27.TextGrid")
f9=r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\WAV\F_free_ninho\gs_27.wav"
f10=r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\WAV\F_nose_ninho\gc_27.wav"


#%% Fun????o que retorna o sinal e a frequ??ncia de amostragem do fonema,
#das vozes hiponasalada e normal 

def feminino(tg1, tg2, f1, f2, tier_name):
    #Fonema a analisar
    phoneme='6~'
   
    #Sinal e frequ??ncia de amostragem do fonema na voz normal
    tgn = tg1.tierDict['annotations'].find(phoneme)
    x, fs = librosa.load(f1)
    #Normaliza????o do sinal
    x=x/(max(abs(x)))
    #Retirar o valor m??dio
    x=x-np.mean(x)
    for idx,k in enumerate(tgn):
        sini = int(tg1.tierDict[tier_name].entryList[k].start*fs)
        send = int(tg1.tierDict[tier_name].entryList[k].end*fs)
        s = x[sini:send]
        sinalFree=s
        srFree=fs
    
    #Sinal e frequ??ncia de amostragem do fonema na voz hiponasalada
    tgn = tg2.tierDict['annotations'].find(phoneme)
    x, fs = librosa.load(f2)
    #Normaliza????o do sinal
    x=x/(max(abs(x)))
    #Retirar o valor m??dio
    x=x-np.mean(x)
    for idx,k in enumerate(tgn):
        sini = int(tg2.tierDict[tier_name].entryList[k].start*fs)
        send = int(tg2.tierDict[tier_name].entryList[k].end*fs)
        s = x[sini:send]
        sinalNose=s
        srNose=fs
        
    return phoneme, sinalFree, srFree, sinalNose, srNose




#%% MFCC E DTW

phoneme, sinalFree1, srFree1, sinalNose1, srNose1=feminino(tg1, tg2, f1, f2, tier_name)
phoneme, sinalFree2, srFree2, sinalNose2, srNose2=feminino(tg3, tg4, f3, f4, tier_name)
phoneme, sinalFree3, srFree3, sinalNose3, srNose3=feminino(tg5, tg6, f5, f6, tier_name)
phoneme, sinalFree4, srFree4, sinalNose4, srNose4=feminino(tg7, tg8, f7, f8, tier_name)
phoneme, sinalFree5, srFree5, sinalNose5, srNose5=feminino(tg9, tg10, f9, f10, tier_name)

#C??lculo dos MFCC para a voz normal e hiponasalada
mfcc1 = librosa.feature.mfcc(sinalFree5, srFree5)
mfcc2 = librosa.feature.mfcc(sinalNose5, srNose5)

#Gr??fico dos MFCC, para a voz normal
fig, ax = plt.subplots()
img = librosa.display.specshow(mfcc1, x_axis='time', ax=ax)
fig.colorbar(img, ax=ax)
ax.set_xlabel('Tempo (s)')
ax.set(title='MFCC ({}) Sem obstru????o nasal'.format(phoneme))

#Gr??fico dos MFCC, para a voz hiponasalada
fig, ax = plt.subplots()
img = librosa.display.specshow(mfcc2, x_axis='time', ax=ax)
fig.colorbar(img, ax=ax)
ax.set_xlabel('Tempo (s)')
ax.set(title='MFCC ({}) Com obstru????o nasal'.format(phoneme))

# Dist??ncia normalizada entre os dois fonemas
dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
print ('Dist??ncia normalizada entre os dois fonemas:', dist)
