# Import necessary modules for threading, GUI, speech recognition, image processing, and file handling.
import threading
import tkinter as Tk
import speech_recognition as sr
from PIL import Image, ImageTk, ImageEnhance
import os

# Define a class for the Speech-to-Sign application.
class speech_to_sign_app:
    def __init__(self, root):
        # Initialize the main window with the given root (Tkinter root object).
        self.root = root
        self.root.title("Speech to Sign App")  # Set the title of the window.
        self.root.geometry("900x650+500+200")  # Set the size and position of the window.
        self.root.configure(background="#f0f4f7")  # Set the background color.

        # Initialize variables to hold the recognized query and running status.
        self.query = ""
        self.run = False

        # Create a label with instructions for the user.
        self.label = Tk.Label(self.root, text="Press the button and start speaking", font=("Helvetica", 18, "bold"), bg="#f0f4f7", fg="#333")
        self.label.pack(pady=20)  # Add padding and pack it into the window.

        # Create a start button to begin speech recognition.
        self.start_button = Tk.Button(self.root, text="Start", font=("Helvetica", 16), bg="#4CAF50", fg="white", command=self.listen_thread, padx=15, pady=10)
        self.start_button.pack(pady=20)  # Add padding and pack the button.

        # Label to show the listening status.
        self.listening_label = Tk.Label(self.root, text="", font=("Helvetica", 16), bg="#f0f4f7", fg="red")
        self.listening_label.pack(pady=20)  # Add padding and pack the label.

        # Frame to contain the conversation text area.
        self.chat_frame = Tk.Frame(self.root, bg="#f0f4f7")
        self.chat_frame.pack(side=Tk.LEFT, padx=20, pady=10)  # Add padding and pack the frame on the left side.

        # Scrollbar for the text area.
        self.scrollbar = Tk.Scrollbar(self.chat_frame)
        self.scrollbar.pack(side=Tk.RIGHT, fill=Tk.Y)  # Attach the scrollbar to the right side.

        # Text widget for displaying recognized text.
        self.text_widget = Tk.Text(self.chat_frame, wrap=Tk.WORD, yscrollcommand=self.scrollbar.set, height=20, width=40, bg="#e6f2ff", fg="#333", font=("Helvetica", 14), bd=2, relief="solid")
        self.text_widget.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=True)  # Pack the text widget inside the chat frame.

        # Link the scrollbar to the text widget.
        self.scrollbar.config(command=self.text_widget.yview)

        # Label to display sign language images.
        self.image_label = Tk.Label(self.root, bg="#f0f4f7", bd=2, relief="solid")
        self.image_label.pack(side=Tk.RIGHT, padx=20, pady=20)  # Pack the image label on the right side.

    # Start the listening function in a separate thread.
    def listen_thread(self):
        thread_listen = threading.Thread(target=self.listen)  # Create a thread for the listen function.
        if self.run is False:  # Only start if not already running.
            thread_listen.start()  # Start the thread.

    # Function to handle speech recognition.
    def listen(self):
        self.run = True  # Set running status to True.
        self.start_button.config(state=Tk.DISABLED)  # Disable the start button while listening.
        self.start_button.config(bg="red")  # Change button color to indicate it is active.
        self.listening_label.config(text="Listening....")  # Update label to show listening status.

        recognizer = sr.Recognizer()  # Create a speech recognizer object.
        with sr.Microphone() as source:  # Use the microphone as the source.
            print("Listening....")
            recognizer.pause_threshold = 1  # Pause before recognizing the next part of speech.
            audio = recognizer.listen(source)  # Capture audio from the microphone.

        try:
            print('Recognizing...')
            self.query = recognizer.recognize_google(audio, language='en-in')  # Recognize the speech using Google's recognizer.
            print(self.query)  # Print the recognized query.

        except Exception as e:
            print(e)  # Print any errors encountered during recognition.

        # If there is a recognized query, display it in the text widget and translate it to sign language.
        if self.query:
            self.text_widget.insert(Tk.END, f"{self.query}\n")  # Add the text followed by a newline in the text widget.
            self.text_widget.see(Tk.END)  # Scroll to the end to see the latest text.
            self.display_sign_language()  # Call the function to display corresponding sign language images.

        # Reset the listening status and button after processing.
        self.listening_label.config(text="")
        self.query = ""
        self.run = False
        self.start_button.config(bg="#4CAF50")
        self.start_button.config(state=Tk.NORMAL)

    # Function to display sign language images corresponding to recognized words or characters.
    def display_sign_language(self):
        self.listening_label.config(text="Translating.....")  # Update status to show translation in progress.
        image_dir = "C:\C progs\FINAL PROJ\My_model\speechtosign\image"  # Directory where images are stored.

        # Check if an image for the whole word exists and display it if found.
        word_image_path = os.path.join(image_dir, f"{self.query}.png")
        if os.path.exists(word_image_path):
            self.fade_in_image(word_image_path)  # Display the word image with fade-in effect.
            self.clear_image_label()  # Clear the label after displaying.
            return

        # Split the query into words and process each word.
        words = self.query.upper().split()

        # Iterate over each word.
        for word in words:
            word_image_path = os.path.join(image_dir, f"{word}.png")  # Check for image of the entire word.
            if os.path.exists(word_image_path):
                self.fade_in_image(word_image_path)  # Display the word image if found.
            else:
                # If word image is not found, split it into characters and look for individual character images.
                characters = [char for char in word if char.isalpha() or char.isdigit()]  # Extract alphabetic or numeric characters.
                for char in characters:
                    image_path = os.path.join(image_dir, f"{char}.png")  # Path for each character image.
                    if os.path.exists(image_path):
                        self.fade_in_image(image_path)  # Display character image with fade-in effect.
                    else:
                        print(f"No image found for {char}")  # Print a message if an image is not found.

        # Clear the image label after processing all words.
        self.clear_image_label()

    # Function to create a fade-in effect for displaying images.
    def fade_in_image(self, image_path):
        # Load the image from the specified path.
        image = Image.open(image_path)
        image = image.resize((300, 300))  # Resize the image to 300x300 pixels.

        # Create a sequence of images with gradually increasing brightness.
        for i in range(1, 11):
            enhancer = ImageEnhance.Brightness(image)  # Create an enhancer for brightness.
            enhanced_image = enhancer.enhance(i / 10)  # Gradually increase brightness.
            photo = ImageTk.PhotoImage(enhanced_image)  # Convert to Tkinter-compatible image.
            self.image_label.config(image=photo)  # Set the label to show the image.
            self.image_label.image = photo  # Keep a reference to prevent garbage collection.
            self.root.update_idletasks()  # Update the display immediately.
            self.root.after(15)  # Pause briefly for the fade-in effect.

        # Display the final fully bright image.
        self.image_label.config(image=photo)
        self.image_label.image = photo
        self.root.update_idletasks()
        self.root.after(1000)  # Keep the final image displayed for 1 second.

    # Function to clear the image from the label.
    def clear_image_label(self):
        self.image_label.config(image='')  # Set the image label to empty.
        self.image_label.image = None  # Remove the reference to the image.
        self.root.update_idletasks()  # Update the display immediately.

# Create the main Tkinter window and run the application.
root = Tk.Tk()
speech_to_sign_app(root)  # Instantiate the app with the root window.
root.mainloop()  # Start the Tkinter event loop to run the app.
