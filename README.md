# 📨 MAIL CLASSIFICATION (SPAM / HAM Email)

Cette application web **Streamlit** classe automatiquement les emails en **HAM** (messages légitimes) ou **SPAM** en utilisant la technologie NLP des **Embeddings** pour comprendre le contexte global des phrases. Les données sémantiques sont ensuite traitées par un modèle de **Régression Logistique** pour garantir une détection précise et éviter les faux positifs des filtres classiques.

---

## 📁 Structure du Projet

L'architecture des scripts Python est organisée de la manière suivante :

* `mail_data.csv` : Le jeu de données initial contenant les emails bruts et leurs étiquettes (Ham/Spam).
* `Tfidf.py` : Le script gérant l'approche statistique de référence (TF-IDF).
* `generate_embeddings.py` : Le script d'extraction sémantique qui calcule la représentation vectorielle des messages.
* `mail_embeddings.npy` : La matrice binaire contenant les vecteurs pré-calculés pour optimiser les performances.
* `app_embedding.py` : L'application web principale développée avec **Streamlit** et utilisant la **Régression Logistique**.

---

## 🧠 Notre Approche : L'Intelligence des Embeddings & Régression Logistique

Pour dépasser la simple analyse par mots-clés qui fait beaucoup d'erreurs, notre solution intègre une chaîne de traitement moderne :

1. **Extraction Sémantique (Embedding) :** Utilisation de la bibliothèque `SentenceTransformer` avec le modèle `all-MiniLM-L6-v2`. Il traduit le texte brut de chaque email en un **vecteur dense à 384 dimensions** (une coordonnée mathématique unique basée sur le sens du message).
2. **Classification (Régression Logistique) :** À partir de ces 384 dimensions sémantiques, le modèle de **Régression Logistique** trace une frontière de décision optimale pour séparer efficacement les emails légitimes (**HAM**) des courriels indésirables (**SPAM**).

---

## ⚔️ Démonstration : Le Duel du Contexte

Notre modèle résout efficacement les faux positifs générés par les filtres classiques (comme TF-IDF) :

* **Cas d'un Email HAM :** *"Can you win the football match tomorrow? Free entry for everyone."*
    * *Filtre classique (TF-IDF) :* Bloque le message et le classe à tort en **SPAM** à cause du mot "Free".
    * *Notre modèle (Embedding + Régression Logistique) :* Capte le contexte amical et sportif. Il le classe correctement en **HAM**.
* **Cas d'un Email SPAM :** *"Free tickets for the World Cup final! Click here to claim your prize now."*
    * *Résultat :* Détecté immédiatement comme **SPAM**, car l'intention marketing et d'urgence (*"Click here"*, *"Prize"*) est repérée géométriquement par le classifieur.

---

## 🛠️ Technologies Utilisées

* **Python** (v3.9+)
* **Streamlit** (Interface Web de test)
* **SentenceTransformers** (Modèles d'Embedding NLP)
* **Scikit-Learn** (Modèle de Régression Logistique)
* **NumPy** (Gestion de la matrice de données `.npy`)

---

## 💻 Installation et Lancement

### 1. Cloner le dépôt
```bash
git clone [https://github.com/houdasoultana23-beep/MAIL_CLASSIFICATION-SPAM---HAM-Email-.git](https://github.com/houdasoultana23-beep/MAIL_CLASSIFICATION-SPAM---HAM-Email-.git)
cd MAIL_CLASSIFICATION-SPAM---HAM-Email-
'''
### 2. Installer les dépendances nécessaires
```bash
pip install streamlit sentence-transformers scikit-learn numpy
'''
3. Exécuter l'application Streamlit
```bash
streamlit run app_embedding.py
'''
