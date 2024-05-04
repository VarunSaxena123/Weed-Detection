import tkinter as tk
import cv2
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from your_model_module import YourModelClass   # Import your model class from the module where it's defined

# Initialize the Tkinter GUI window
root = tk.Tk()
root.title("Crop and Weed Detection")

# Load your trained PyTorch model
model = YourModelClass()  # Instantiate your model class
model.load_state_dict(torch.load('performing_detection_with_pytorch.pth'))  # Load the saved model weights
model.eval()  # Set the model to evaluation mode

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Function to preprocess the image for the model
def preprocess_image(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Assuming input size of your model is 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Adjust normalization as needed
    ])
    return transform(frame).unsqueeze(0)

# Function to perform crop and weed detection on each frame
def detect_crop_and_weed(frame):
    # Preprocess the frame for the model
    processed_frame = preprocess_image(frame)

    # Perform inference with the model
    with torch.no_grad():
        output = model(processed_frame)
        # Process the model output as needed
        # For example, you can apply a threshold to the output to determine if crop or weed is detected

    return frame  # Placeholder for processed frame

# Function to update the GUI with webcam feed
def update_frame():
    ret, frame = cap.read()
    if ret:
        processed_frame = detect_crop_and_weed(frame)

        # Convert processed frame to format compatible with Tkinter
        processed_frame_tk = ImageTk.PhotoImage(image=Image.fromarray(processed_frame))

        # Update the label with the processed frame
        label.config(image=processed_frame_tk)
        label.image = processed_frame_tk

    root.after(10, update_frame)

# Create a label to display webcam feed
label = tk.Label(root)
label.pack()

# Start updating the frame
update_frame()

# Run the Tkinter GUI main loop
root.mainloop()

# Release the webcam and clean up
cap.release()
cv2.destroyAllWindows()
