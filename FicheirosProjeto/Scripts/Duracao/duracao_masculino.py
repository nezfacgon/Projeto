# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 21:07:46 2021

@author: nezfa
"""

import numpy as np
from praatio import tgio

tier_name = 'annotations'

#%% Ficheiros TextGrid

tg1=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\M_free_ninho\gs_9.TextGrid")
tg2=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\M_nose_ninho\gc_9.TextGrid")

tg3=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\M_free_ninho\gs_12.TextGrid")
tg4=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\M_nose_ninho\gc_12.TextGrid")

tg5=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\M_free_ninho\gs_18.TextGrid")
tg6=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\M_nose_ninho\gc_18.TextGrid")

tg7=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\M_free_ninho\gs_24.TextGrid")
tg8=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\M_nose_ninho\gc_24.TextGrid")

tg9=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\M_free_ninho\gs_30.TextGrid")
tg10=tgio.openTextgrid(r"C:\Users\nezfa\OneDrive\Ambiente de Trabalho\Disciplinas\2Semestre\PROEST\Dados\Projeto\TextGrid\M_nose_ninho\gc_30.TextGrid")



#%% Função que permite imprimir a duração de um fonema
def get_durations(phoneme, tg1, tg2):
    
        print('A processar /',phoneme,'/...')
        tgn = tg1.tierDict['annotations'].find(phoneme)
        dur = np.zeros(len(tgn))
        for idx,k in enumerate(tgn):
            dur[idx] = tg1.tierDict[tier_name].entryList[k].end - tg1.tierDict[tier_name].entryList[k].start
        print('FREE ({}): Duração média (ms): {}'.format(idx+1,np.round(np.mean(dur)*1000,1)))
       
        tgn = tg2.tierDict['annotations'].find(phoneme)
        dur = np.zeros(len(tgn))
        for idx,k in enumerate(tgn):
            dur[idx] = tg2.tierDict[tier_name].entryList[k].end - tg2.tierDict[tier_name].entryList[k].start
        print('NOSE ({}): Duração média (ms): {}'.format(idx+1,np.round(np.mean(dur)*1000,1)))


#%%Imprimir duração de tempo dos fonemas
get_durations('o~', tg1, tg2)
get_durations('bri~', tg1, tg2)
get_durations('u~', tg1, tg2)
get_durations('ni', tg1, tg2)
get_durations('Ju', tg1, tg2)
get_durations('6~', tg1, tg2)
get_durations('J6S', tg1, tg2)
get_durations('e~', tg1, tg2)
get_durations('mu', tg1, tg2)
