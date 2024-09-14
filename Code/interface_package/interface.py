import customtkinter

# Créer l'application
app = customtkinter.CTk()
app.title("Fenêtre de test")
app.geometry('400x150')

label = customtkinter.CTkLabel(app, text="Voici la fenêtre de test de Gwenn")
label.pack(pady=10)

# Fonction pour ouvrir la boite de dialogue et afficher le texte
def ouvrir_boite():
    dialog = customtkinter.CTkInputDialog(text="Insère des données", title="Boîte de dialogue")
    texte_saisi = dialog.get_input()  # Récupérer le texte saisi depuis la boite de dialogue
    if texte_saisi:  # Si du texte est saisi
        my_label.configure(text=f"Texte saisi : {texte_saisi}")  # Afficher le texte saisi

# Créer le bouton pour envoyer les données
my_button = customtkinter.CTkButton(app, text="Send", command=ouvrir_boite)
my_button.pack(pady=20)

# Créer le label pour afficher le texte saisi
my_label = customtkinter.CTkLabel(app, text="")
my_label.pack(pady=0)

# Lancer l'application
app.mainloop()