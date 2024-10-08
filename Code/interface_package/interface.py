import customtkinter
from Code.input.titanic.prep_data import test_user_input

def app_interface(mlp, testdata):
    # Créer l'application
    app = customtkinter.CTk()
    app.title("MLP Model Training : Titanic Dataset")
    app.geometry('600x400')  # Taille de la fenêtre

    label = customtkinter.CTkLabel(app, text="MLP Model Training : for Titanic Dataset")
    label.pack(pady=10)

    # Créer le label pour afficher le statut
    my_label = customtkinter.CTkLabel(app, text="")
    my_label.pack(pady=10)

    # Fonction pour tester le modèle (bouton pour tester le modèle)
    def tester_modele():
        my_label.configure(text="Le modèle est en cours de test...")  # Indication du test en cours
        result = testdata(mlp)  # Appel de la fonction de test avec le modèle
        my_label.configure(text=f"Résultat du test : {result:.2f}% de précision")  # Affiche le float avec 2 décimales

    # Créer le bouton pour tester le modèle
    train_button = customtkinter.CTkButton(app, text="Tester le modèle", command=tester_modele)
    train_button.pack(pady=10)  # Assurer que le bouton de test soit visible

    # Fonction pour entrer des données et les afficher
    def entrer_donnees():
        input_window = customtkinter.CTkToplevel(app)
        input_window.title("Entrer des données")
        input_window.geometry('400x400')

        # Labels et champs d'entrée pour chaque donnée
        fields = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        entries = {}

        for field in fields:
            input_label = customtkinter.CTkLabel(input_window, text=f"{field} :")
            input_label.pack(pady=5)

            if field == 'Sex':
                # Utiliser un menu déroulant pour 'Sex'
                input_entry = customtkinter.CTkComboBox(input_window, values=["male", "female"])
                input_entry.pack(pady=5)
            elif field == 'Pclass':
                # Utiliser un menu déroulant pour 'Pclass'
                input_entry = customtkinter.CTkComboBox(input_window, values=["1", "2", "3"])
                input_entry.pack(pady=5)
            elif field == 'Embarked':
                # Utiliser un menu déroulant pour 'Embarked'
                input_entry = customtkinter.CTkComboBox(input_window, values=["C", "Q", "S"])
                input_entry.pack(pady=5)
            else:
                input_entry = customtkinter.CTkEntry(input_window)
                input_entry.pack(pady=5)

            entries[field] = input_entry

        # Fonction pour afficher les données saisies et la prédiction
        def afficher_donnees():
            data = {
                'Pclass': entries['Pclass'].get(),
                'Sex': entries['Sex'].get(),
                'Age': entries['Age'].get(),
                'SibSp': entries['SibSp'].get(),
                'Parch': entries['Parch'].get(),
                'Fare': entries['Fare'].get(),
                'Embarked': entries['Embarked'].get()
            }

            # Créer une chaîne de texte avec les données saisies
            result_text = (f"Données saisies :\nPclass : {data['Pclass']}\n"
                           f"Sex : {data['Sex']}\nAge : {data['Age']}\n"
                           f"SibSp : {data['SibSp']}\nParch : {data['Parch']}\n"
                           f"Fare : {data['Fare']}\nEmbarked : {data['Embarked']}\n")

            # Préparer les données pour la prédiction
            user_input = {
                'Pclass': int(data['Pclass']),
                'Sex': 0 if data['Sex'] == 'male' else 1,  # Convertir 'male'/'female' en 0/1
                'Age': float(data['Age']),
                'SibSp': int(data['SibSp']),
                'Parch': int(data['Parch']),
                'Fare': float(data['Fare']),
                'Embarked': data['Embarked']  # À gérer selon votre modèle
            }

            # Appel à la fonction test_user_input
            prediction_accuracy = test_user_input(mlp, user_input)  # Ajustez cette fonction pour renvoyer la prédiction
            if prediction_accuracy == 0:
                result_text += "Prédiction : Passenger didn't survive"
            else:
                result_text += "Prédiction : Passenger survived"
            my_label.configure(text=result_text)  # Affiche dans le label principal

        # Bouton pour soumettre et afficher les données
        test_button = customtkinter.CTkButton(input_window, text="Afficher les données et prédiction", command=afficher_donnees)
        test_button.pack(pady=20)

    # Créer le bouton pour entrer des données
    data_button = customtkinter.CTkButton(app, text="Entrer des données", command=entrer_donnees)
    data_button.pack(pady=10)

    # Lancer l'application
    app.mainloop()
