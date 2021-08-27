#from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
#from keras.engine.topology import Network
#from keras.layers import *
#from keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
import tensorflow.keras.backend as K
from PIL import Image # zaimportowanie modułu potrzebnego na laptopie
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


# stworzenie kodera
def stworzenie_kodera(rozmiar_wejściowy):
	wejście_S = Input(shape=(rozmiar_wejściowy))
	wejście_C = Input(shape=(rozmiar_wejściowy))

	# siec przygotowująca
	x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='selu', name='conv_prep0_3x3')(input_S)
	x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='selu', name='conv_prep0_4x4')(input_S)
	x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='selu', name='conv_prep0_5x5')(input_S)
	x = concatenate([x3, x4, x5])
    
	x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='selu', name='conv_prep1_3x3')(x)
	x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='selu', name='conv_prep1_4x4')(x)
	x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='selu', name='conv_prep1_5x5')(x)
	x = concatenate([x3, x4, x5])

	x = concatenate([wejście_C, x])

	# siec ukrywająca
	x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='selu', name='conv_hid0_3x3')(x)
	x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='selu', name='conv_hid0_4x4')(x)
	x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='selu', name='conv_hid0_5x5')(x)
	x = concatenate([x3, x4, x5])
    
	x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='selu', name='conv_hid1_3x3')(x)
	x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='selu', name='conv_hid1_4x4')(x)
	x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='selu', name='conv_hid1_5x5')(x)
	x = concatenate([x3, x4, x5])
    
	x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='selu', name='conv_hid2_3x3')(x)
	x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='selu', name='conv_hid2_4x4')(x)
	x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='selu', name='conv_hid2_5x5')(x)
	x = concatenate([x3, x4, x5])
    
	x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='selu', name='conv_hid3_3x3')(x)
	x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='selu', name='conv_hid3_4x4')(x)
	x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='selu', name='conv_hid3_5x5')(x)
	x = concatenate([x3, x4, x5])
    
	x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='selu', name='conv_hid4_3x3')(x)
	x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='selu', name='conv_hid4_4x4')(x)
	x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='selu', name='conv_hid5_5x5')(x)
	x = concatenate([x3, x4, x5])
    
	wyjściowy_C_zmod = Conv2D(3, (3, 3), strides = (1, 1), padding='same', activation='selu', name='wyjściowy_C')(x)

	return Model(wejścia=[wejście_S, wejście_C],
				 wyjścia=wyjściowy_C_zmod,
				 nazwa = 'koder')

def stworzenie_dekodera(rozmiar_wejściowy, stała=False):

	# siec ekstrakcyjna
	wejscie_ekstrakcja = Input(rozmiar=(rozmiar_wejściowy))

	# dodanie szumu gausowskiego
	zaszumione_wejscie = GaussianNoise(0.01, nazwa='zaszumiona_okładka')(wejscie_ekstrakcja)

	x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='selu', name='conv_rev0_3x3')(zaszumione_wejscie)
	x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='selu', name='conv_rev0_4x4')(zaszumione_wejscie)
	x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='selu', name='conv_rev0_5x5')(zaszumione_wejscie)
	x = concatenate([x3, x4, x5])
    
	x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='selu', name='conv_rev1_3x3')(x)
	x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='selu', name='conv_rev1_4x4')(x)
	x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='selu', name='conv_rev1_5x5')(x)
	x = concatenate([x3, x4, x5])
    
	x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='selu', name='conv_rev2_3x3')(x)
	x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='selu', name='conv_rev2_4x4')(x)
	x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='selu', name='conv_rev2_5x5')(x)
	x = concatenate([x3, x4, x5])
    
	x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='selu', name='conv_rev3_3x3')(x)
	x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='selu', name='conv_rev3_4x4')(x)
	x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='selu', name='conv_rev3_5x5')(x)
	x = concatenate([x3, x4, x5])
    
	x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='selu', name='conv_rev4_3x3')(x)
	x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='selu', name='conv_rev4_4x4')(x)
	x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='selu', name='conv_rev5_5x5')(x)
	x = concatenate([x3, x4, x5])
    
	wyjściowy_S_zmod = Conv2D(3, (3, 3), strides = (1, 1), padding='same', activation='selu', name='wyjście_S')(x)

	if not stała:
		return Model(wejścia=wejscie_ekstrakcja,
					 wyjścia=wyjściowy_S_zmod,
					 nazwa = 'Dekoder')
	else:
		return Siec(wejścia=wejscie_ekstrakcja,
					wyjścia=wyjściowy_S_zmod,
					nazwa = 'Naprawiony dekoder')

# cały model
def stworzenie_modelu(rozmiar_wejściowy):
	wejście_S = Input(shape=(rozmiar_wejściowy))
	wejście_C= Input(shape=(rozmiar_wejściowy))

	koder = stworzenie_kodera(rozmiar_wejściowy)

	dekoder = stworzenie_dekodera(rozmiar_wejściowy)
	dekoder.compile(optimizer='adam', loss=rev_loss)
	dekoder.trainable = False

	wyjściowy_C_zmod = koder([wejście_S, wejście_C])
	wyjściowy_S_zmod = dekoder(wyjściowy_C_zmod)

	autokoder = Model(wejścia=[wejście_S, wejście_C],
					  wyjścia=concatenate([wyjściowy_S_zmod, wyjściowy_C_zmod]))
	autokoder.compile(optimizer='adam',loss=full_loss)

	return koder, dekoder, autokoder

def współczynnik_uczenia(epoki):
	if epoki < 200:
		return 0.001
	elif epoki < 300:
		return 0.0003

import wandb
wandb.init(project='steganografia', entity='maciej-mazuro')
sweep_config = {
	'method': 'random',
	'metric': {
	  'name': 'rev_loss',
	  'goal': 'minimize'
	},
	'parameters': {

		'lr':{
			'values':[0.001]
		},
		'activation':{
			'values':['relu']
		}
	}
}

sweep_id = wandb.sweep(sweep_config)