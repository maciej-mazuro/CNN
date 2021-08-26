#from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
#from keras.engine.topology import Network
#from keras.layers import *
#from keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

import numpy as np
import os
import random
import scipy.misc
#from tqdm import *

# Wskazanie ścieżki do zbiorów danych zawierający obrazy
zbiór_danych = "./tiny-imagenet-200"
zbiór_treningowy = os.path.join(zbiór_danych, "train")	# zbiór danych treningowych
zbiór_testowy = os.path.join(zbiór_danych, "test") # zbiór danych testowych

rozmiar_obrazów = (64, 64) # określenie rozmiaru pojedyńczego obrazka

def wczytanie_zbiorów(liczba_obrazów_w_zbiorze_treningowym=10, liczba_obrazów_testowych=500):
	X_treningowe = [] # stworzenie macierzy obrazów treningowych
	X_testowe = [] # stworzenie macierzy obrazów testowych

	# stworzenie zbioru treningowego
	for c in os.listdir(zbiór_treningowy):
		c_zbiór = os.path.join(zbiór_treningowy, c, 'images')
		c_obrazy = os.listdir(c_zbiór)
		random.shuffle(c_obrazy)
		for nazwa_obrazu in c_obrazy[0:liczba_obrazów_w_zbiorze_treningowym]:
			obraz = image.load_img(os.path.join(c_zbiór, nazwa_obrazu))
			x = image.img_to_array(obraz)
			X_treningowe.append(x)
	random.shuffle(X_treningowe)

	# stworzenie zbioru testowego
	testowy_zbiór = os.path.join(zbiór_testowy, 'images')
	obrazy_testowe = os.listdir(testowy_zbiór)
	random.shuffle(obrazy_testowe)
	for nazwa_obrazu in obrazy_testowe[0:liczba_obrazów_testowych]:
		obraz = image.load_img(os.path.join(testowy_zbiór, nazwa_obrazu))
		x = image.img_to_array(obraz)
		X_testowe.append(x)

	return np.array(X_treningowe), np.array(X_testowe)

# wczytanie zbioru danych
X_treningowe_oryginał, X_testowe_oryginał = wczytanie_zbiorów()

# normalizacja wektorów obrazowych
X_treningowe = X_treningowe_oryginał/255.
X_testowe = X_testowe_oryginał/255.

# wyświetlenie statystyki
print("Liczba przykładów treningowych = " + str(X_treningowe.shape[0]))
print("Liczba przykładów testowych = " + str(X_treningowe.shape[0]))
print("Wymiary zbioru treningowego: " + str(X_treningowe.shape))

# podział zbioru danych - połowa użyta jest jako ukrywane obrazki
# druga połowa jako obrazy okładki

# S : ukrywany obraz(ang. secret image)
tajny_obraz = X_treningowe[0:X_treningowe.shape[0] // 2]

# C : obraz okładki(ang. cover image)
okładka = X_treningowe[X_treningowe.shape[0] // 2:]

# wyświetlenie obrazów ze zbioru treningowego
zestawienie=plt.figure(figsize=(8, 8))
kolumny = 4
rzędy = 5
for i in range(1, kolumny*rzędy +1):
	# losowe obrazki
	indeks_obrazu = np.random.choice(X_treningowe.shape[0])
	zestawienie.add_subplot(rzędy, kolumny, i)
	plt.imshow(X_treningowe[indeks_obrazu])
plt.show()

import wandb
beta = 1.0

# funkcja straty sieci neuronowej zajmującej się ekstrakcją
def strata_ekstrakcji(s_prawdziwy, s_predykcja):
	# wzór na stratę
	return beta * K.sum(K.square(s_prawdziwy - s_predykcja))

# funkcja straty dla całego modelu
def strata_modelu(y_prawdziwy, y_predykcja):
	# wzór na stratę
	s_prawdziwy, c_prawdziwy = y_prawdziwy[...,0:3], y_prawdziwy[...,3:6]
	s_predykcja, c_predykcja = y_predykcja[...,0:3], y_predykcja[...,3:6]

	s_strata = rev_loss(s_prawdziwy, s_predykcja)
	c_strata = K.sum(K.square(c_prawdziwy - c_predykcja))

	return s_strata + c_strata