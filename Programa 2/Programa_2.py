# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 22:54:20 2022

@author: MarioLap
"""

#Exportar datos


#EJERCICIO1

# t-test for independent samples
from math import sqrt
import numpy as np
from numpy import mean
from scipy.stats import sem
from scipy.stats import t
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from numpy.random import randn
import pandas as pd


# generate two independent samples
lista1 = 5 * randn(100) + 50
lista2 = 5 * randn(100) + 50

#Exportar datos
new_list = {"Exp1": list(lista1), "Exp2": list(lista2)}
df = pd.DataFrame(new_list)
writer = pd.ExcelWriter("test.xlsx", engine="xlsxwriter")
df.to_excel(writer, sheet_name="welcome", index=False)
writer.save()

#Importar
df2 = pd.read_excel("test.xlsx")

Experimento_a = df2['Exp1']
Experimento_b = df2['Exp2']

#T-student para muestras independientes
def ttest(Experimento_a, Experimento_b, alpha):
   
# calcular medias
    mean1, mean2 = mean(Experimento_a), mean(Experimento_b)
   
# calcular erros estandar
    se1, se2 = sem(Experimento_a), sem(Experimento_b)
   
# error estándar en la diferencia entre las muestras
    sed = sqrt(se1**2.0 + se2**2.0)
   
# calcular el estadistico t
    t_stat = (mean1 - mean2) / sed
   
# grados de libertad
    df = len(Experimento_a) + len(Experimento_b) - 2
   
# calcular el valor critico
    cv = t.ppf(1.0 - alpha, df)
   
# calcular el p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
   
    return t_stat, df, cv, p
 

# Agsinar alfa
alpha = 0.05
 
#Programa principal
val = 0

while val != 5:
    print("0. Mostrar valores")
    print("1. Aplicar t-student")
    print("2. Correlacion de Pearson")
    print("3. Correlacion de Spearman")
    print("4. Graficar diagrama de dispersion")
    print("5. Salir")
 
    val= int(input())
   
    if(val==0):
        print("Lista 1: ",Experimento_a)
        print("Lista 2: ",Experimento_b)
             
    if(val==1):
        t_stat, df, cv, p = ttest(Experimento_a, Experimento_b, alpha)
        print('t=%.3f, df=%d, cv=%.3f, p=%.3f\n' % (t_stat, df, cv, p))
        # interpret via p-value
        if p > alpha:
            print('No hay diferencia significativa entre las medias.\n')
        else:
            print('Hay diferencia significativa entre las medias.\n')
 
    if(val==2):
        #Correlacion de Pearson
        corr, p1 = pearsonr(Experimento_a, Experimento_b)
        print('Correlacion de Pearson: %.3f\n' % corr)
   
    if(val==3):
        #Correlacion de Spearman
        rho, p2 = spearmanr (Experimento_a, Experimento_b)
        print('Correlacion de Spearman: %.3f\n' % rho)
   
    if(val==4):
        #Grafico de dispercion
        fig, ax = plt.subplots()
        ax.scatter(Experimento_a,Experimento_b)
        plt.plot(Experimento_a, Experimento_b, 'o')
        m,b1=np.polyfit(Experimento_a, Experimento_b, 1)
        pend= m*Experimento_a+b1
       
        plt.plot(Experimento_a, pend)
        plt.pause(0.2)