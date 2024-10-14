# Installation du projet 
Clonez le répo dans un dossier de votre choix.

Si vous êtes sur MAC, rendez vous dans le dossier preprocess. Sinon ignorez cette étape.

Pour installer les bibliothèques nécessaire au fonctionnement du projet (dans un environnement virtuel de préférence), lancez la commande suivante :
```
pip install -r requirements.txt
```

Vérifier que la bibliothèque transformers est bien installée avec la commande suivante :
```
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

La sortie devrait être la suivante avec un score similaire mais pas forcement exact a celui-là :
```
[{'label': 'POSITIVE', 'score': 0.9998704195022583}]
``` 

Une fois ces étapes validées l'installation est terminée et vous pouvez désormais lancer votre application.

Pour lancer le "back" du projet, rendez-vous dans le dossier "test" et lancez la commande :
```
uvicorn backend:app
```
Pour lancer le "front", lancez la commande dans un terminal séparé :
```
streamlit run frontend.py
```
