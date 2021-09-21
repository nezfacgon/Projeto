# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 10:30:07 2021

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

tg1=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\M_free_ninho\gs_9.TextGrid")
tg2=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\M_nose_ninho\gc_9.TextGrid")
f1=r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\WAV\M_free_ninho\gs_9.wav"
f2=r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\WAV\M_nose_ninho\gc_9.wav"

tg3=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\M_free_ninho\gs_12.TextGrid")
tg4=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\M_nose_ninho\gc_12.TextGrid")
f3=r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\WAV\M_free_ninho\gs_12.wav"
f4=r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\WAV\M_nose_ninho\gc_12.wav"

tg5=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\M_free_ninho\gs_18.TextGrid")
tg6=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\M_nose_ninho\gc_18.TextGrid")
f5=r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\WAV\M_free_ninho\gs_18.wav"
f6=r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\WAV\M_nose_ninho\gc_18.wav"

tg7=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\M_free_ninho\gs_24.TextGrid")
tg8=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\M_nose_ninho\gc_24.TextGrid")
f7=r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\WAV\M_free_ninho\gs_24.wav"
f8=r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\WAV\M_nose_ninho\gc_24.wav"

tg9=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\M_free_ninho\gs_30.TextGrid")
tg10=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\M_nose_ninho\gc_30.TextGrid")
f9=r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\WAV\M_free_ninho\gs_30.wav"
f10=r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\WAV\M_nose_ninho\gc_30.wav"


#%% Função que retorna o sinal e a frequência de amostragem do fonema,
#das vozes hiponasalada e normal 

def masculino(tg1, tg2, f1, f2, tier_name):
    #Fonema a analisar
    phoneme='6~'
   
   #Sinal e frequência de amostragem do fonema na voz normal
    tgn = tg1.tierDict['annotations'].find(phoneme)
    x, fs = librosa.load(f1)
    #Normalização do sinal
    x=x/(max(abs(x)))
    #Retirar o valor médio
    x=x-np.mean(x)
    for idx,k in enumerate(tgn):
        sini = int(tg1.tierDict[tier_name].entryList[k].start*fs)
        send = int(tg1.tierDict[tier_name].entryList[k].end*fs)
        s = x[sini:send]
        sinalFree=s
        srFree=fs

    #Sinal e frequência de amostragem do fonema na voz hiponasalada
    tgn = tg2.tierDict['annotations'].find(phoneme)
    x, fs = librosa.load(f2)
    #Normalização do sinal
    x=x/(max(abs(x)))
    #Retirar o valor médio
    x=x-np.mean(x)
    for idx,k in enumerate(tgn):
        sini = int(tg2.tierDict[tier_name].entryList[k].start*fs)
        send = int(tg2.tierDict[tier_name].entryList[k].end*fs)
        s = x[sini:send]
        sinalNose=s
        srNose=fs

    return phoneme, sinalFree, srFree, sinalNose, srNose




#%% MFCC E DTW

phoneme, sinalFree1, srFree1, sinalNose1, srNose1=masculino(tg1, tg2, f1, f2, tier_name)
phoneme, sinalFree2, srFree2, sinalNose2, srNose2=masculino(tg3, tg4, f3, f4, tier_name)
phoneme, sinalFree3, srFree3, sinalNose3, srNose3=masculino(tg5, tg6, f5, f6, tier_name)
phoneme, sinalFree4, srFree4, sinalNose4, srNose4=masculino(tg7, tg8, f7, f8, tier_name)
phoneme, sinalFree5, srFree5, sinalNose5, srNose5=masculino(tg9, tg10, f9, f10, tier_name)

#Cálculo dos MFCC para a voz normal e hiponasalada
mfcc1 = librosa.feature.mfcc(sinalFree5, srFree5)
mfcc2 = librosa.feature.mfcc(sinalNose5, srNose5)

#Gráfico dos MFCC, para a voz normal
fig, ax = plt.subplots()
img = librosa.display.specshow(mfcc1, x_axis='time', ax=ax)
fig.colorbar(img, ax=ax)
ax.set_xlabel('Tempo (s)')
ax.set(title='MFCC ({}) Sem obstrução nasal'.format(phoneme))

#Gráfico dos MFCC, para a voz hiponasalada
fig, ax = plt.subplots()
img = librosa.display.specshow(mfcc2, x_axis='time', ax=ax)
fig.colorbar(img, ax=ax)
ax.set_xlabel('Tempo (s)')
ax.set(title='MFCC ({}) Com obstrução nasal'.format(phoneme))

# Distância normalizada entre os dois fonemas
dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
print ('Distância normalizada entre os dois fonemas:', dist)
