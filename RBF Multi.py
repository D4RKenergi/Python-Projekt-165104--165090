import numpy as np
import multiprocessing as mp
import keras


path = 'C:/Users/Kacper/Downloads/STUDIA/semestr 6/Sztuczna Inteligencja/Projekt/'
labels = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
n_cat = len(labels)

train_X = np.load("Dane/emnist-balanced-train-30-X.npy")
train_Y = np.load("Dane/emnist-balanced-train-30-Y.npy")
test_X = np.load("Dane/emnist-balanced-test-10-X.npy")
test_Y = np.load("Dane/emnist-balanced-test-10-Y.npy")


#Euklidesowa odlełość. Pierwiastek z sumy kwadratów odległości.
def euc_dist(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return np.sqrt(sum)

def kmeans(X, k):
    #Wybieramy k losowych elementów jako początki centroid
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]
    
    #Funkcja ma 100 powtórzeń na znalezienie optymalnego rozkładu
    for i in range(100):
        #Lista [[][]...[]] przynależnych elementów do centroid
        clusters = [[] for _ in range(len(centroids))] 
        
        for x_i, x in enumerate(X):  #Przejdź przez każdy element
            distances_list = []
            for c_i, c in enumerate(centroids): #Przejdz przez centroidy
                distances_list.append(euc_dist(c, x)) #Odległość centroidy od zdjęcia

            #Element przypisywany jest do najbliższego klastra / centroidy
            clusters[int(np.argmin(distances_list))].append(x)

        centroids_old = centroids.copy() # Kopiowanie do porównania
        
        # Kalkuluje średnią pozycję centroid na podstawie zdjęć przypisanych do klastra
        centroids = [np.mean(c, axis=0) for c in clusters]

        diff = np.abs(np.sum(centroids_old) - np.sum(centroids)) # Porównanie z poprzednią iteracją

        if diff == 0: #Jeśli jest optymalne rozmieszczenie to zakończ
            break
    
    return centroids # Zwraca centroidy oraz odchylenie standardowe

class RBF:

    def __init__(self, X, Y, TX, TY, k):
        self.TrX = X #Zdjęcia do trenowania
        self.TrY = Y #Klasy zdjęć do trenowania
        self.TeX = TX #Zdjęcia do testowania
        self.TeY = TY #Klasy zdjęć do testowania
        self.k = k #Ilość centroid / neuronów

    #Zwraca wartość funkcji radialnej
    def get_rbf(self, x, c, B):
        distance = euc_dist(x, c) #Odległość elementu od centroida
        return np.exp(-(distance**2) / (2*B**2)) #Funckja radialna

    #Zwraca listę rbf dla podanych elementów
    def get_rbf_matrix(self, X, centroids, B):
        #Dla każdego elementu zwraca wartość rbf od wszystkich centroid
        return np.array([[self.get_rbf(x, c, B) for c in centroids] for x in X])
    
    #Wyznacza centroidy dla elementów
    def find_kmeans(self):
        self.centroids = kmeans(self.TrX, self.k) 

    #Wyznaczanie wag
    def fit(self, B):
        #Zwraca macierz rbf dla elementów trenujących
        RBF_X = self.get_rbf_matrix(self.TrX, self.centroids, B) 
        #Oblicza wagi na podstawie listy rbf i odpowiadających jej elementom klas
        self.W = np.linalg.pinv(RBF_X.T @ RBF_X) @ RBF_X.T @ keras.utils.np_utils.to_categorical(self.TrY) 

    #Testowanie sieci
    def test(self, B):
        #Zwraca macierz rbf dla elementów testowych
        RBF_TX = self.get_rbf_matrix(self.TeX, self.centroids, B) 
        self.PredTeY = RBF_TX @ self.W #Mnożenie rbf z wagami
        
        #Wybiera neurony z największą sumą jako wynik
        self.PredTeY = np.array([np.argmax(x) for x in self.PredTeY]) 

        #Oblicza różnicę między klasami przewidzianymi a prawdziwymi
        diff = self.PredTeY - self.TeY 
        #Oblicza stosunek prawidłowych trafień
        self.Accuracy = len(np.where(diff == 0)[0]) / len(diff)
        print('Accuracy: ', self.Accuracy)

        #Tworzenie tabeli lasyfikacji
        self.accuracy_table = np.zeros((47,47))
        #Przechodzi przez wszystkie elementy testujące
        for i, y in enumerate(self.TeY):
            #Zwiększa numerację w odpowiednim miejscu tabeli
            self.accuracy_table[y, self.PredTeY[i]] += 1
            #Dodaje element po drugiej stronie dla symetryczności
            if y != self.PredTeY[i]:
                self.accuracy_table[self.PredTeY[i], y] += 1

def Fit(B,cl):
    RBF_CLASSIFIER = RBF(train_X, train_Y, test_X, test_Y, cl)
    RBF_CLASSIFIER.centroids = np.load('Output/30/centroids ' + str(cl) + ' - 30.npy')
    print("Fiting: cl" + str(cl) + " B" + str(B))
    RBF_CLASSIFIER.fit(B)
    RBF_CLASSIFIER.test(B)
    np.save('Output/30/weights ' + str(cl) + ' - 30 v3 B' + str(B) + " acc" + str(RBF_CLASSIFIER.Accuracy) + ".npy", RBF_CLASSIFIER.W)
    np.save('Output/30/acc_table ' + str(cl) + ' - 30 v3 B' + str(B) + ".npy", RBF_CLASSIFIER.accuracy_table)

def Test(B, file):
    RBF_CLASSIFIER = RBF(train_X, train_Y, test_X, test_Y, num_of_classes=47, k=700, std_from_clusters=False)
    RBF_CLASSIFIER.centroids = np.load('Output/30/centroids 10 - 30.npy')
    #RBF_CLASSIFIER.std_list = np.load('Output/30/std_list 700 - 30.npy')
    RBF_CLASSIFIER.w = np.load('Output/30/' + file)
    print("Testing: " + str(B))
    RBF_CLASSIFIER.test(B)
    np.save('Output/30/acc_table 700 - 30 v3 B' + str(B) + ".npy", RBF_CLASSIFIER.accuracy_table)

if __name__ == '__main__':
    with mp.Pool(5) as pool:
        a = [0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20, 40, 75, 100, 125, 150]
        c = [10] * 20
        b = ["weights 700 - 30 v3 B0.25 acc0.09787234042553192.npy", "weights 700 - 30 v3 B0.5 acc0.11914893617021277.npy", "weights 700 - 30 v3 B0.75 acc0.3829787234042553.npy",
        "weights 700 - 30 v3 B1 acc0.4085106382978723.npy", "weights 700 - 30 v3 B2 acc0.44468085106382976.npy", "weights 700 - 30 v3 B3 acc0.5276595744680851.npy",
        "weights 700 - 30 v3 B4 acc0.5617021276595745.npy", "weights 700 - 30 v3 B5 acc0.5851063829787234.npy", "weights 700 - 30 v3 B6 acc0.5702127659574469.npy"]
        pool.starmap(Fit, zip(a,c))
        #pool.starmap(Test, zip(a, b))