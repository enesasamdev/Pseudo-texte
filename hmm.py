import random as rd
import numpy as np
import matplotlib.pyplot as plt

# NETTOYAGE DU TEXTE""
""" Ici, on récupère le texte des misérables dans un fichier 
puis on enlève tous les caractères spéciaux par des espaces, 
les retours à la ligne,
les accents par leurs lettres sans accent,
on supprime les doubles/triples espaces créés,
Enfin on met toutes les lettres en majuscule. """

fichier = open('miserables.txt', 'r')
ls = []
for l in fichier:
    ls.append(l[:-1]) # Ajout de la ligne sans son \n "invisible" dans un tableau 

txtEntier  = " ".join(ls) # On transforme la liste en chaîne de caractères séparés par des espaces

txtEntier = txtEntier.lower() # On met tout en minuscule

txtEntier = txtEntier.translate(txtEntier.maketrans("àâçéèêëîïñôùûü","aaceeeeiinouuu")) # On supprime tous les accents dans le texte
txtEntier = txtEntier.replace('æ','ae') # On dédouble les ligatures pour ae
txtEntier = txtEntier.replace('œ','oe') # On dédouble les ligatures pour oe

for c in '0123456789:;,.?!—_°[]()*#«»"<>':
    txtEntier = txtEntier.replace(c, '') # On supprime tous les caractères spéciaux

""" Pour le changement ci-dessous, il est important de laisser cette ordre de remplacement car la liste 
ajoutera au fil des changements des espaces en plus """
for c in ['-','/','\'', '\t','    ','   ','  ']:
    txtEntier = txtEntier.replace(c, ' ') # On remplace les caractères de la liste par un espace

txtEntier = txtEntier.upper() # On met tous le texte en majuscule


# MODELE DE MARKOV

lettres = [] # Correspond à la liste de lettres de l'alphabet + le caractère espace
for i in range(26):
    lettres.append(chr(i+65)) # On ajoute chaque lettre de l'alphabet à partir de son code ascii
lettres += [' '] # Ajout de l'espace

dico = {} # Dictionnaire pour associer une lettre (et l'espace) à un nombre
for i in range(27):
    dico[lettres[i]] = i # 0 pour A, 1 pour B, ...

lsTxtToNb = [] # liste correspondant au texte (sauf que chaque caractères est remplacé par son nombre du dictionnaire)
for lettre in txtEntier:
    lsTxtToNb.append(dico[lettre])


# LETTRES
mOrdre1 = np.array([[0]*27]*27) # Matrice vide de taille 27*27 (pour les 26 lettres et l'espace)
for i in range(len(lsTxtToNb)-1):
    mOrdre1[lsTxtToNb[i],lsTxtToNb[i+1]] +=1 # On ajoute 1 pour chaque position dans la matrice afin de calculer la fréquence

transpose = np.transpose(mOrdre1) # On fait la transposé de la matrice
somme = sum(transpose) # On calcule la somme de la transposé
ls_fq = somme/sum(somme) # Liste de la fréquence d'apparition de chaque caractère du dictionnaire dans le texte

print("\nLa fréquence des lettres et de l'espace dans le texte :\n".upper())
for i in range(len(lettres)):
    print('\t\t',lettres[i],':',round(ls_fq[i]*100,2),'%')
print("\n")


# DIGRAMMES
mOrdre2 = mOrdre1[:-2,:-2]
print("Les digrammes apparaissant le plus de fois dans le texte :\n".upper())
for i in range(len(lettres[:-2])):
    for j in range(len(lettres[:-2])):
        if mOrdre2[i][j] > 5000: # On affiche seulement ceux qui apparraissent plus de 5000 fois, sinon il y en aurait trop
            print('\t\t',lettres[i] + lettres[j],':',mOrdre2[i][j])
print("\n")


# TRIGRAMMES
mOrdre2 = np.array([[[0]*27]*27]*27) # Matrice vide de taille 27*27*27 (pour les 26 lettres et l'espace)
for i in range(len(lsTxtToNb)-2):
    mOrdre2[lsTxtToNb[i],lsTxtToNb[i+1],lsTxtToNb[i+2]] +=1 # On ajoute 1 pour chaque position dans la matrice afin de calculer la fréquence

mOrdre3 = mOrdre2[:-2,:-2,:-2]
print("Les trigrammes apparaissant le plus de fois dans le texte :\n".upper())
for i in range(len(lettres[:-2])):
    for j in range(len(lettres[:-2])):
        for k in range(len(lettres[:-2])):
            if mOrdre3[i][j][k] > 1500: # On affiche seulement ceux qui apparraissent plus de 1500 fois, sinon il y en aurait trop
                print('\t\t',lettres[i] +lettres[j] + lettres[k],':',mOrdre3[i][j][k])
print()

matrice = np.array([[[[0]*27]*27]*27]*27)
for i in range(len(lsTxtToNb)-3):
    matrice[lsTxtToNb[i],lsTxtToNb[i+1],lsTxtToNb[i+2],lsTxtToNb[i+3]] +=1


# Texte modèle de Markov d'ordre 1
txtMarkov1 = rd.choices(lettres, somme)[0] # Choisit aléatoirement une lettre en fonction des statistiques
for i in range(2000):
    txtMarkov1 = txtMarkov1 + rd.choices(lettres,mOrdre1[dico[txtMarkov1[-1]]])[0] # On génère un texte de 2000 caractères
print("\n\tTexte obtenu à partir du modèle de Markov d'ordre 1 :\n")
print(txtMarkov1)

""" On repète la même avec plus de choix en fonction de l'ordre du modèle de Markov """

# Texte modèle de Markov d'ordre 2
choix = rd.choices(lettres,somme)
txtMarkov2 = choix[0] + rd.choices(lettres,mOrdre1[dico[choix[-1]]])[0]
for i in range(2000):
    txtMarkov2 = txtMarkov2 + rd.choices(lettres,mOrdre2[dico[txtMarkov2[-2]],dico[txtMarkov2[-1]]])[0]
print("\n\n\tTexte obtenu à partir du modèle de Markov d'ordre 2 :\n")
print(txtMarkov2)

# Texte modèle de Markov d'ordre 3
choix = rd.choices(lettres,somme)
choix2 = rd.choices(lettres,mOrdre1[dico[choix[-1]]])
txtMarkov3 = choix[0] + choix2[0] + rd.choices(lettres,mOrdre2[dico[choix[-1]],dico[choix2[-1]]])[0]
for i in range(2000):
    txtMarkov3 = txtMarkov3 + rd.choices(lettres,matrice[dico[txtMarkov3[-3]],dico[txtMarkov3[-2]],dico[txtMarkov3[-1]]])[0]
print("\n\n\tTexte obtenu à partir du modèle de Markov d'ordre 3 :\n")
print(txtMarkov3)
print()


def distribution_len_mots(fichier,ordre):
    """ Affiche sous forme d'histogramme la présence
    des mots en fonction de leurs longueurs """
    taille_mots = []
    liste_mots = fichier.split(' ')
    for mot in liste_mots:
        taille_mots.append(len(mot))
    fig1 = plt.figure()
    titre = "Taille des mots avec Markov d'ordre "+str(ordre)
    plt.title(titre, fontsize=20)
    plt.xlabel("Taille des mots")
    plt.ylabel("Nombre de mots")
    plt.hist(taille_mots, bins = max(taille_mots), facecolor='orange', alpha=1, edgecolor = "black", density=True)
    plt.savefig(titre)


distribution_len_mots(txtMarkov1,1)
distribution_len_mots(txtMarkov2,2)
distribution_len_mots(txtMarkov3,3)