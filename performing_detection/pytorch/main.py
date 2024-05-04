# import tkinter as tk
# from tkinter import filedialog
# import matplotlib.pyplot as plt
# from darknet import Darknet
# from utils import *
# import cv2

# # Define paths and parameters
# cfg_file = '/Users/atharv07/Desktop/crop_weed/Crop_and_weed_detection-master/performing_detection/data/cfg/crop_weed.cfg'
# weight_file = '/Users/atharv07/Desktop/crop_weed/Crop_and_weed_detection-master/performing_detection/data/weights/crop_weed_detection.weights'
# namesfile = '/Users/atharv07/Desktop/crop_weed/Crop_and_weed_detection-master/performing_detection/data/names/obj.names'
# m = Darknet(cfg_file)
# m.load_weights(weight_file)
# class_names = load_class_names(namesfile)

# # Function to perform detection and display result
# def perform_detection(image_path, iou_thresh=0.4, nms_thresh=0.6):
#     # Load the image
#     img = cv2.imread(image_path)
#     # Convert the image to RGB
#     original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # We resize the image to the input width and height of the first layer of the network.    
#     resized_image = cv2.resize(original_image, (m.width, m.height))
#     # Detect objects in the image
#     boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)
#     # Print the objects found and the confidence level
#     print_objects(boxes, class_names)
#     # Plot the image with bounding boxes and corresponding object class labels
#     plot_boxes(original_image, boxes, class_names, plot_labels=True)
#     plt.show()

# # Function to handle file selection and initiate detection
# def select_file_and_detect():
#     file_path = filedialog.askopenfilename()
#     if file_path:
#         perform_detection(file_path)

# # Create main application window
# root = tk.Tk()
# root.title("Crop and Weed Detection")

# # Create a button to select image file
# select_button = tk.Button(root, text="Select Image", command=select_file_and_detect)
# select_button.pack(pady=10)

# # Start the GUI event loop
# root.mainloop()



# import tkinter as tk
# import cv2
# from PIL import Image, ImageTk
# from darknet import Darknet
# from utils import *

# # Initialize the Darknet model and load weights
# cfg_file = '/Users/atharv07/Desktop/crop_weed/Crop_and_weed_detection-master/performing_detection/data/cfg/crop_weed.cfg'
# weight_file = '/Users/atharv07/Desktop/crop_weed/Crop_and_weed_detection-master/performing_detection/data/weights/crop_weed_detection.weights'
# namesfile = '/Users/atharv07/Desktop/crop_weed/Crop_and_weed_detection-master/performing_detection/data/names/obj.names'
# m = Darknet(cfg_file)
# m.load_weights(weight_file)
# class_names = load_class_names(namesfile)

# # Initialize webcam capture
# cap = cv2.VideoCapture(0)
# current_frame = None  # Placeholder for the current frame from the webcam

# # Function to perform detection on the captured image
# def detect_and_display():
#     global current_frame
#     if current_frame is not None:
#         # Convert the frame to RGB format
#         frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
#         # Resize the frame to match model input size
#         resized_frame = cv2.resize(frame_rgb, (m.width, m.height))
#         # Perform object detection
#         boxes = detect_objects(m, resized_frame, iou_thresh=0.4, nms_thresh=0.6)
#         # Print detected objects and confidence levels
#         print_objects(boxes, class_names)
#         # Plot the frame with bounding boxes and labels
#         plot_boxes(frame_rgb, boxes, class_names, plot_labels=True)
#         # Update the GUI with the processed frame
#         update_frame(frame_rgb)

# # Function to update the GUI with processed frame
# def update_frame(frame):
#     # Convert the frame to ImageTk format compatible with Tkinter
#     img_tk = ImageTk.PhotoImage(image=Image.fromarray(frame))
#     # Update the label with the processed frame
#     label.config(image=img_tk)
#     label.image = img_tk

# # Function to capture a photo from the webcam and perform detection
# def capture_photo():
#     global current_frame
#     ret, frame = cap.read()
#     if ret:
#         current_frame = frame
#         detect_and_display()

# # Create main application window
# root = tk.Tk()
# root.title("Crop and Weed Detection")

# # Create a label to display webcam feed
# label = tk.Label(root)
# label.pack(pady=10)

# # Create a button to capture photo
# capture_button = tk.Button(root, text="Capture Photo", command=capture_photo)
# capture_button.pack(pady=10)

# # Function to update the GUI with live webcam feed
# def update_live_preview():
#     ret, frame = cap.read()
#     if ret:
#         # Convert the frame to RGB format
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # Convert the frame to ImageTk format compatible with Tkinter
#         img_tk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
#         # Update the label with the live webcam feed
#         label.config(image=img_tk)
#         label.image = img_tk
#     # Call the function again after a delay (in milliseconds)
#     label.after(10, update_live_preview)

# # Start updating the live webcam preview
# update_live_preview()

# # Run the Tkinter GUI main loop
# root.mainloop()

# # Release the webcam
# cap.release()



import tkinter as tk
import cv2
from PIL import Image, ImageTk
from darknet import Darknet
from utils import *

# Initialize the Darknet model and load weights
cfg_file = '/Users/atharv07/Desktop/crop_weed/Crop_and_weed_detection-master/performing_detection/data/cfg/crop_weed.cfg'
weight_file = '/Users/atharv07/Desktop/crop_weed/Crop_and_weed_detection-master/performing_detection/data/weights/crop_weed_detection.weights'
namesfile = '/Users/atharv07/Desktop/crop_weed/Crop_and_weed_detection-master/performing_detection/data/names/obj.names'
m = Darknet(cfg_file)
m.load_weights(weight_file)
class_names = load_class_names(namesfile)

# Initialize webcam capture
cap = cv2.VideoCapture(0)
current_frame = None  # Placeholder for the current frame from the webcam

# Function to perform detection on the captured image
def detect_and_display():
    global current_frame
    if current_frame is not None:
        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        # Resize the frame to match model input size
        resized_frame = cv2.resize(frame_rgb, (m.width, m.height))
        # Perform object detection
        boxes = detect_objects(m, resized_frame, iou_thresh=0.4, nms_thresh=0.6)
        # Print detected objects and confidence levels
        print_objects(boxes, class_names)
        # Plot the frame with bounding boxes and labels
        plot_boxes(frame_rgb, boxes, class_names, plot_labels=True)
        # Update the GUI with the processed frame (resized for preview)
        update_frame(cv2.resize(frame_rgb, (640, 480)))  # Increased preview size

# Function to update the GUI with processed frame
def update_frame(frame):
    # Convert the frame to ImageTk format compatible with Tkinter
    img_tk = ImageTk.PhotoImage(image=Image.fromarray(frame))
    # Update the label with the processed frame
    label.config(image=img_tk)
    label.image = img_tk

# Function to capture a photo from the webcam and perform detection
def capture_photo():
    global current_frame
    ret, frame = cap.read()
    if ret:
        current_frame = frame
        detect_and_display()

# Create main application window
root = tk.Tk()
root.title("Crop and Weed Detection")

# Create a label to display webcam feed with a slightly larger preview size
label = tk.Label(root)
label.pack(pady=10)
label.config(width=640, height=480)  # Set the width and height for the label (slightly larger preview)

# Create a button to capture photo
capture_button = tk.Button(root, text="Capture Photo", command=capture_photo)
capture_button.pack(pady=10)

# Function to update the GUI with live webcam feed
def update_live_preview():
    ret, frame = cap.read()
    if ret:
        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to ImageTk format compatible with Tkinter
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        # Update the label with the live webcam feed
        label.config(image=img_tk)
        label.image = img_tk
    # Call the function again after a delay (in milliseconds)
    label.after(10, update_live_preview)

# Start updating the live webcam preview
update_live_preview()

# Run the Tkinter GUI main loop
root.mainloop()

# Release the webcam
cap.release()
