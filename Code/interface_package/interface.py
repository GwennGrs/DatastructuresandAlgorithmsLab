import customtkinter

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
        my_label.configure(text=f"Résultat du test : {result}")  # Mise à jour du label après le test

    # Créer le bouton pour tester le modèle
    train_button = customtkinter.CTkButton(app, text="Tester le modèle", command=tester_modele)
    train_button.pack(pady=10)  # Assurer que le bouton de test soit visible

    # Fonction pour entrer des données et les afficher
    def entrer_donnees():
        input_window = customtkinter.CTkToplevel(app)
        input_window.title("Entrer des données")
        input_window.geometry('400x400')

        # Labels et champs d'entrée pour chaque donnée
        fields = ['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        entries = {}

        for field in fields:
            input_label = customtkinter.CTkLabel(input_window, text=f"{field} :")
            input_label.pack(pady=5)
            input_entry = customtkinter.CTkEntry(input_window)
            input_entry.pack(pady=5)
            entries[field] = input_entry

        # Fonction pour afficher les données saisies
        def afficher_donnees():
            data = {
                'Sex': entries['Sex'].get(),
                'Age': entries['Age'].get(),
                'SibSp': entries['SibSp'].get(),
                'Parch': entries['Parch'].get(),
                'Fare': entries['Fare'].get(),
                'Embarked': entries['Embarked'].get()
            }

            # Créer une chaîne de texte avec les données saisies
            result_text = f"Données saisies :\nSex : {data['Sex']}\nAge : {data['Age']}\nSibSp : {data['SibSp']}\nParch : {data['Parch']}\nFare : {data['Fare']}\nEmbarked : {data['Embarked']}"
            my_label.configure(text=result_text)  # Affiche dans le label principal

        # Bouton pour soumettre et afficher les données
        test_button = customtkinter.CTkButton(input_window, text="Afficher les données", command=afficher_donnees)
        test_button.pack(pady=20)

    # Créer le bouton pour entrer des données
    data_button = customtkinter.CTkButton(app, text="Entrer des données", command=entrer_donnees)
    data_button.pack(pady=10)

    # Lancer l'application
    app.mainloop()
