# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 14:27:51 2021

@author: nezfa
"""

import numpy as np
from scipy.fft import fft,fftfreq
import librosa
import matplotlib.pyplot as plt
from praatio import tgio

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
        
#%% Função que retorna o fonema a analisar, a amplitude espetral e a frequência do mesmo,
#das vozes hiponasalada e normal 
def spectrum(tg1, tg2, f1, f2, tier_name):
    #Fonema a analisar
    phoneme='6~'    

    #Determinação da amplitude espetral do fonema, na voz normal
    tgn = tg1.tierDict['annotations'].find(phoneme)
    x, fs = librosa.load(f1)
    #Normalização do sinal
    x=x/(max(abs(x)))
    #Retirar o valor médio
    x=x-np.mean(x)
    nfft=1024
    Y=np.zeros((len(tgn),nfft//2))
    for idx,k in enumerate(tgn):
        sini = int(tg1.tierDict[tier_name].entryList[k].start*fs)
        send = int(tg1.tierDict[tier_name].entryList[k].end*fs)
        s = x[sini:send]
        temp = fft(s, nfft)
        Y[idx,:] = 2/nfft*np.abs(temp[0:nfft//2])
    w=fftfreq(nfft,1/fs)[:nfft//2]
    Y=20*np.log10(Y)
    
    #Determinação da amplitude espetral do fonema, na voz hiponasalada
    tgn = tg2.tierDict['annotations'].find(phoneme)
    x, fs = librosa.load(f2)
    #Normalização do sinal
    x=x/(max(abs(x)))
    #Retirar o valor médio
    x=x-np.mean(x)
    nfft=1024
    S=np.zeros((len(tgn),nfft//2))
    for idx,k in enumerate(tgn):
        sini = int(tg2.tierDict[tier_name].entryList[k].start*fs)
        send = int(tg2.tierDict[tier_name].entryList[k].end*fs)
        s = x[sini:send]
        temp = fft(s, nfft)
        S[idx,:]= 2/nfft*np.abs(temp[0:nfft//2])
    w=fftfreq(nfft,1/fs)[:nfft//2]
    S=20*np.log10(S)
    
    return S, w,Y, phoneme


#%% Determinação de espetros e amplitudes
S1,w,Y1,phoneme=spectrum(tg1, tg2, f1, f2, tier_name)
S2,w,Y2,phoneme=spectrum(tg3, tg4, f3, f4, tier_name)
S3,w,Y3,phoneme=spectrum(tg5, tg6, f5, f6, tier_name)
S4,w,Y4,phoneme=spectrum(tg7, tg8, f7, f8, tier_name)
S5,w,Y5,phoneme=spectrum(tg9, tg10, f9, f10, tier_name)

#Média das amplitudes espetrais na voz normal
print("Free(1):{}".format(np.mean(Y1)))
print("Free(2):{}".format(np.mean(Y2)))
print("Free(3):{}".format(np.mean(Y3)))
print("Free(4):{}".format(np.mean(Y4)))
print("Free(5):{}".format(np.mean(Y5)))

#Média das amplitudes espetrais na voz hiponasalada
print("\nNose(1):{}".format(np.mean(S1)))
print("Nose(2):{}".format(np.mean(S2)))
print("Nose(3):{}".format(np.mean(S3)))
print("Nose(4):{}".format(np.mean(S4)))
print("Nose(5):{}".format(np.mean(S5)))

#Criação de vetores contendo as amplitudes espetrais, do mesmo fonema, em diferentes sinais
S = np.array([S1[0], S2[0], S3[0], S4[0], S5[0]])
Y = np.array([Y1[0], Y2[0], Y3[0], Y4[0], Y5[0]])

#Gráfico da média das amplitudes espetrais em função da frequência, do fonema, para a voz normal
plt.plot(w,np.mean(Y,0))
plt.fill_between(w, np.min(Y,0), np.max(Y,0), alpha=0.2)
plt.title('Sem obstrução nasal ({})'.format(phoneme))
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude|S| (dB)')
plt.grid()
plt.show()

#Gráfico da média das amplitudes espetrais em função da frequência, do fonema, para a voz hiponasalada
plt.plot(w,np.mean(S,0),'-r')
plt.fill_between(w, np.min(S,0), np.max(S,0), alpha=0.2,color=(1,0,0))
plt.title('Com obstrução nasal ({})'.format(phoneme))
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude|S| (dB)')
plt.grid()
plt.show()



#%% Cálculo da distância euclidiana entre o espetro da voz normal e da hiponasalada

Y=np.mean(Y,0)
S=np.mean(S,0)


#Distância euclidiana
n=len(S)
lista_dif=[]
for i in range(n):
    
    dif=(S[i]-Y[i])**2
    lista_dif.append(dif)

soma=np.sum(lista_dif)
sqrt=np.sqrt(soma)

print("\nDistância: %s" %sqrt)

#%% Cálculo da média da amplitude espetral
mediaFree=np.mean(Y)
mediaNose=np.mean(S)
print('\nMédia da amplitude espetral (Free): {}'.format(mediaFree))
print('Média da amplitude espetral (Nose): {}'.format(mediaNose))

