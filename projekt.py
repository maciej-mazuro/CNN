import os

# Wskazanie ścieżki do zbiorów danych zawierający obrazy
zbiór_danych = "./tiny-imagenet-200"
zbiór_treningowy = os.path.join(zbiór_danych, "train")	# zbiór danych treningowych
zbiór_testowy = os.path.join(zbiór_danych, "test") # zbiór danych testowych

rozmiar_obrazów = (64, 64) # określenie rozmiaru pojedyńczego obrazka

def wczytanie_zbiorów(liczba_obrazów_w_zbiorze_treningowym=10, liczba_obrazów_testowych=500):
	X_treningowe = []
	X_testowe = []

	# stworzenie zbioru treningowego
	for c in os.listdir(zbiór_treningowy):
		c_zbiór = os.path.join(zbiór_treningowy, c, 'obrazy')
		c_obrazy = os.listdir(c_zbiór)
		random.shuffle(c_obrazy)
		for nazwa_obrazu in c_obrazy[0:liczba_obrazów_w_zbiorze_treningowym]:
			obraz = image.load.img(os.path.join(c_zbiór, nazwa_obrazu))
			x = image.img_to_array(obraz)
			X_treningowe.append(x)
	random.shuffle(X_treningowe)

	# stworzenie zbioru testowego
	testowy_zbiór = os.path.join(zbiór_testowy, 'obrazy')
	obrazy_testowe = os.listdir(testowy_zbiór)
	random.shuffle(obrazy_testowe)
	for nazwa_obrazu in obrazy_testowe[0:liczba_obrazów_testowych]:
		obraz = image.load_img(os.path.join(testowy_zbiór, nazwa_obrazu))
		x = image.img_to_array(obraz)
		X_testowe.append(x)

	return np.array(X_treningowe), np.array(X_testowe)