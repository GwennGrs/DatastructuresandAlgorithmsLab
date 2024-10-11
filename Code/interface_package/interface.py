import customtkinter
from Code.input.titanic.prep_data import test_user_input

#Test the model
def tester_modele(mlp, testdata, my_label):
    """
    Test the machine learning model with the provided test data and display the result in a label.
    This function updates the label to indicate that the model is being tested, calls the test function,
    and then updates the label with the test result.
    
    Args:
        mlp (object): The machine learning model to be tested.
        testdata (function): A function that takes the model as input and returns the test accuracy.
        my_label (tkinter.Label): The label widget where the test result will be displayed.
    
    Returns:
        None
    """
    my_label.configure(text="The model is being tested...")  # Indicate that the test is in progress
    result = testdata(mlp)  # Call the test function with the model
    my_label.configure(text=f"Test result: {result:.2f}% accuracy")  # Display the test result with 2 decimal places

def afficher_donnees(entries, mlp, my_label, input_window):
    """
    Display entered data and prediction result in a label, then close the input window.
    This function retrieves user input from the provided entries, formats the data for display,
    prepares it for prediction, and then calls a prediction function. The result is displayed
    in the provided label, and the input window is closed.
    Args:
        entries (dict): A dictionary containing the user input fields. Each key corresponds to a field name,
                        and the value is a Tkinter Entry widget.
        mlp (object): The machine learning model used for prediction.
        my_label (tkinter.Label): The label widget where the result will be displayed.
        input_window (tkinter.Toplevel): The input window that will be closed after displaying the result.
    Returns:
        None
    """
    data = {
        'Pclass': entries['Pclass'].get(),
        'Sex': entries['Sex'].get(),
        'Age': entries['Age'].get(),
        'SibSp': entries['SibSp'].get(),
        'Parch': entries['Parch'].get(),
        'Fare': entries['Fare'].get(),
        'Embarked': entries['Embarked'].get()
    }

    # Create a text string with the entered data
    result_text = (f"Entered data:\nPclass: {data['Pclass']}\n"
                   f"Sex: {data['Sex']}\nAge: {data['Age']}\n"
                   f"SibSp: {data['SibSp']}\nParch: {data['Parch']}\n"
                   f"Fare: {data['Fare']}\nEmbarked: {data['Embarked']}\n")

    # Prepare the data for prediction
    user_input = {
        'Pclass': int(data['Pclass']),
        'Sex': 0 if data['Sex'] == 'male' else 1,  # Convert 'male'/'female' to 0/1
        'Age': float(data['Age']),
        'SibSp': int(data['SibSp']),
        'Parch': int(data['Parch']),
        'Fare': float(data['Fare']),
        'Embarked': data['Embarked']  # To be handled according to your model
    }

    # Call the test_user_input function
    prediction_accuracy = test_user_input(mlp, user_input)  # Adjust this function to return the prediction
    if prediction_accuracy == 0:
        result_text += "Prediction: Passenger didn't survive"
    else:
        result_text += "Prediction: Passenger survived"
    my_label.configure(text=result_text)  # Display in the main label
    input_window.destroy()  # Ferme la fenêtre après l'affichage

# Enter data
def entrer_donnees(app, mlp, my_label):
    """
    Opens a new window to enter data for prediction.
    Parameters:
    app (customtkinter.CTk): The main application window.
    mlp (object): The machine learning model used for prediction.
    my_label (customtkinter.CTkLabel): The label to display the prediction result.
    The function creates a new window with entry fields for the following data:
    - Pclass: Passenger class (dropdown menu with options "1", "2", "3")
    - Sex: Gender (dropdown menu with options "male", "female")
    - Age: Age (text entry)
    - SibSp: Number of siblings/spouses aboard (text entry)
    - Parch: Number of parents/children aboard (text entry)
    - Fare: Ticket fare (text entry)
    - Embarked: Port of embarkation (dropdown menu with options "C", "Q", "S")
    A button is provided to submit the entered data and display the prediction result.
    """
    input_window = customtkinter.CTkToplevel(app)
    input_window.title("Enter data")
    input_window.geometry('400x400')

    # Labels and entry fields for each data
    fields = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    entries = {}

    for field in fields:
        input_label = customtkinter.CTkLabel(input_window, text=f"{field} :")
        input_label.pack(pady=5)

        if field == 'Sex':
            # Use a dropdown menu for 'Sex'
            input_entry = customtkinter.CTkComboBox(input_window, values=["male", "female"])
            input_entry.pack(pady=5)
        elif field == 'Pclass':
            # Use a dropdown menu for 'Pclass'
            input_entry = customtkinter.CTkComboBox(input_window, values=["1", "2", "3"])
            input_entry.pack(pady=5)
        elif field == 'Embarked':
            # Use a dropdown menu for 'Embarked'
            input_entry = customtkinter.CTkComboBox(input_window, values=["C", "Q", "S"])
            input_entry.pack(pady=5)
        else:
            input_entry = customtkinter.CTkEntry(input_window)
            input_entry.pack(pady=5)

        entries[field] = input_entry

    # Button to submit and display the data
    test_button = customtkinter.CTkButton(input_window, text="Display data and prediction", command=lambda: afficher_donnees(entries, mlp, my_label, input_window))
    test_button.pack(pady=20)

# Application interface
def app_interface(mlp, testdata):
    """
    Initializes and runs the graphical user interface for the MLP Model Training on the Titanic Dataset.
    Parameters:
    mlp (object): The trained MLP model to be tested.
    testdata (DataFrame): The test data to be used for model evaluation.
    The interface includes:
    - A title and window size setup.
    - A label displaying the application title.
    - A label to display the status of the model testing.
    - A button to test the model, which triggers the `tester_modele` function.
    - A button to enter data, which triggers the `entrer_donnees` function.
    The application runs in a loop until the user closes the window.
    """
    # Create the application
    app = customtkinter.CTk()
    app.title("MLP Model Training : Titanic Dataset")
    app.geometry('600x400')  # Window size

    label = customtkinter.CTkLabel(app, text="MLP Model Training : for Titanic Dataset")
    label.pack(pady=10)

    # Create the label to display the status
    my_label = customtkinter.CTkLabel(app, text="")
    my_label.pack(pady=10)

    # Create the button to test the model
    train_button = customtkinter.CTkButton(app, text="Test the model", command=lambda: tester_modele(mlp, testdata, my_label))
    train_button.pack(pady=10)  # Ensure the test button is visible

    # Create the button to enter data
    data_button = customtkinter.CTkButton(app, text="Enter data", command=lambda: entrer_donnees(app, mlp, my_label))
    data_button.pack(pady=10)

    # Launch the application
    app.mainloop()
