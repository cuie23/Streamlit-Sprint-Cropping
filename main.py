import cv2
from ultralytics import YOLO
import imageio
import numpy as np
import math
import PIL
from PIL import ImageTk, Image  
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import *
from pynput.mouse import Listener
import streamlit as st

# Sets up letter dictionary
letterCrops = {}

letter_w = 130
letter_h = 212
for i in range (3):
    for j in range (7):
        char = chr(97 + 7*i + j)
        letterCrops.update({char : (105 + j*letter_w + j*63, 100 + i*letter_h, 105 + (j+1)*letter_w + j*63, 100 + (i+1)*letter_h)})

for i in range(1, 6):
    char = chr(117 + i)
    letterCrops.update({char : (100 + (i)*(letter_w+15) + i*48, 100+letter_h*3, 100 + (i+1)*(letter_w+15) + i*48, 100+letter_h*4)})

# Resizes an image 
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

# Slightly expands the crop (since sometimes the detector box can cut off a toe)
def GetBoundingBox(x1, y1, x2, y2, img_w, img_h):
    y_boundary = int(abs(y1-y2)*0.1)
    x_boundary = int((abs(y1-y2)*GUI_width/1000 - abs(x1-x2))/2)

    new_x1 = x1 - x_boundary
    new_y1 = y1 - y_boundary
    new_x2 = x2 + x_boundary 
    new_y2 = y2 + y_boundary

    #returns xyxy coords of adjusted box
    return (max(0, new_x1), max(0, new_y1), min(img_w, new_x2), min(img_h, new_y2))

# If there are 2 or more cropped frames, calculates velocity of person and guesses where 
# they will be based on the frame number of the next crop in case detection fails
def extrapolate_box_x1(frame0, frame1, currFrameNumber, x0, x1):
    frameDiff = abs(frame0-frame1)
    xDiff = abs(x0-x1)
    if (xDiff > 0):
        r = float(frameDiff)/float(xDiff)
        return x1 + int(r*(currFrameNumber-frame1))
    return x1


# Gets a specific letter from the letters.jpg image, returns np image
def getLetter(char):
    global letterCrops

    if (char == ' '):
        output = np.zeros((212, 145, 3))
        output[:, :, :] = 255

        return output

    letters = cv2.imread('letters.jpg')

    output = np.zeros((212, 145, 3))
    output[:, :, :] = 255

    x1, y1, x2, y2 = letterCrops[char]
    output[:abs(y1-y2), :abs(x1-x2)] = letters[y1:y2, x1:x2]  
    return output

# Size Variables
GUI_width = 1200
GUI_height = 750    

img_width = int((GUI_width*2)/5)
img_height = int((GUI_width*2)/5)

# GUI setup
window=tk.Tk()
input_dim = str(GUI_width)+'x'+str(GUI_height)
window.geometry(input_dim)

# Some counter variables to keep track of which frame/how many images have been pasted respectively
frameNumber = IntVar()
frameNumber.set(1)
cropped_frameNumbers = []
cropped_frameX = []
cropNumber = IntVar()
cropNumber.set(0)

# File Variable
currFile = StringVar()
currFile.set('black_screen.mp4')

# Bounding box variables 
box_x1 = IntVar()
box_y1 = IntVar()
box_x2 = IntVar()
box_y2 = IntVar()
box_x1.set(-1)
box_y1.set(-1)
box_x2.set(-1)
box_y2.set(-1)


# Gets video from mp4 file
cap = cv2.VideoCapture(currFile.get())
cap.set(cv2.CAP_PROP_POS_FRAMES, frameNumber.get())
success, img = cap.read()
img = ResizeWithAspectRatio(img, img_width, img_height)

# File Variable
currFile = StringVar()
currFile.set('black_screen.mp4')

# Gets number of frames in video
videoLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

success, frame = cap.read()
model = YOLO('yolov8m.pt')
results = []
results = model(img, conf = 0.3)

# tries to extrapolate the x coords if the detector fails
if (len(cropped_frameX) >= 2):
    box_w = abs(box_x1.get() - box_x2.get())
    box_x1.set(extrapolate_box_x1(cropped_frameNumbers[-1], cropped_frameNumbers[-2], 
                                  frameNumber.get(), cropped_frameX[-1], cropped_frameX[-2]))
    box_x2.set(box_x1.get() + box_w)

# Sets box tensor based on detector
if (len(results) > 0):
    boxes = results[0].boxes
    if(len(boxes) > 0):
        box_tensor = boxes[0].xyxy
        box_x1.set(int(box_tensor[0][0].item()))
        box_y1.set(int(box_tensor[0][1].item()))
        box_x2.set(int(box_tensor[0][2].item()))
        box_y2.set(int(box_tensor[0][3].item()))

# Converting to tkinter Image
image1 = PIL.Image.fromarray(np.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
frame = ImageTk.PhotoImage(image1)

# Putting the frame on screen
videoFrame = Label(window, image = frame)
videoFrame.place(x = int(GUI_width/13.5), y = int(GUI_height/45), anchor = 'nw')

# List of labels that will turn into images when the user decides to add them
pixel = tk.PhotoImage(width=1, height=1)
imgLabel_list = []
img_list = []
for i in range (10):
    l = Label(window, image = pixel, bg = 'white')
    imgLabel_list.append(l)
    img_list.append(pixel)


# Gets nth frame (tkinter img)
def getImage(n):
    global img_width, img_height
    cap = cv2.VideoCapture(currFile.get())
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameNumber.get())
    success, img = cap.read()
    
    img = ResizeWithAspectRatio(img, img_width, img_height)
    image1 = PIL.Image.fromarray(np.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    frame = ImageTk.PhotoImage(image1)

    return frame



# Functions to parse through frames
def incrFrame():
    if (frameNumber.get() < videoLength-1):
        frameNumber.set(frameNumber.get()+1)
        img2 = getImage(frameNumber.get())
        videoFrame.configure(image=img2)
        videoFrame.image = img2

def decrFrame():
    if (frameNumber.get() >= 0):
        frameNumber.set(frameNumber.get()-1)
    img2 = getImage(frameNumber.get())
    videoFrame.configure(image=img2)
    videoFrame.image = img2

def incrFrame10():
    if (frameNumber.get() < videoLength-11):
        frameNumber.set(frameNumber.get()+10)
        img2 = getImage(frameNumber.get())
        videoFrame.configure(image=img2)
        videoFrame.image = img2

def decrFrame10():
    if (frameNumber.get() >= 10):
        frameNumber.set(frameNumber.get()-10)
    img2 = getImage(frameNumber.get())
    videoFrame.configure(image=img2)
    videoFrame.image = img2

# Prevent errors when cropping
def adjustXBounds(n):
    global img_width
    return int(min(img_width, max(0, n)))

def adjustYBounds(n):
    global img_height
    return int(min(img_height, max(0, n)))

# Crops image based on Pose Detection
def cropImage():
    global img_width, img_height, cropped_frameNumbers, cropped_frameX

    # adding frame number of crop for x coordinate extrapolation
    cropped_frameNumbers.append(frameNumber.get())

    cap = cv2.VideoCapture(currFile.get())
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameNumber.get())
    success, img = cap.read()
    img = ResizeWithAspectRatio(img, img_width, img_height)

    # running model
    model = YOLO('yolov8m.pt')
    results = []
    results = model(img, conf = 0.3)
    
    # Sets box tensor based on detector
    if (len(results) > 0):
        boxes = results[0].boxes
        if(len(boxes) > 0):
            box_tensor = boxes[0].xyxy
            box_x1.set(int(box_tensor[0][0].item()))
            box_y1.set(int(box_tensor[0][1].item()))
            box_x2.set(int(box_tensor[0][2].item()))
            box_y2.set(int(box_tensor[0][3].item()))

    # adds padding to detector box + adds a new x coordinate reference for position extrapolation
    x1, y1, x2, y2 = GetBoundingBox(box_x1.get(), box_y1.get(), box_x2.get(), box_y2.get(), img_width, img_height)
    cropped_frameX.append(x1)
    
    # if the detection is successful then return the cropped image
    if (x1 >= 0 and y1 >= 0):
        crop_img = img[y1:y2, x1:x2]
        image1 = PIL.Image.fromarray(np.uint8(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)))
        ratio = float(abs(x1-x2))/float(abs(y1-y2))

        image2 = image1.resize((int(ratio*4.5*GUI_width/30), int(4.5*GUI_width/30)))
        frame = ImageTk.PhotoImage(image2)
        return (0, frame)
    
    # if detection is not successful then return a fail code and a pixel image
    listImg = np.array([[0]])
    img = PIL.Image.fromarray(cv2.cvtColor(listImg, cv2.COLOR_BGR2RGB))
    img2 = ImageTk.PhotoImage(img)
    return (1, img2)


# Places phase image labels
for i in range(2):
    for j in range(5):
        idx = i*5 + j
        xPos = int(GUI_width/10 + j*GUI_width/5)
        yPos =  int(GUI_height*3/5 + i*GUI_height/4)
        imgLabel_list[idx].place(x = xPos, y = yPos, anchor = CENTER)

# Place phase name labels
phaseNameList = ['Toe off', 'Max Vert Pos', 'Strike', 'Touch down', 'Full support']
for i in range (5):
    pNameLabel = Label(window, text = phaseNameList[i], font = ('TkDefaultFont', 15))
    pNameLabel.place(x = int(GUI_width/10 + i*GUI_width/5), y = int(GUI_height*3/7), anchor = CENTER)
    


# Add an image to the board
def pasteImage(): 
    global imgLabel_list
    if(cropNumber.get() < 10):
        success, img = cropImage()
        if (success == 0):
            l = imgLabel_list[cropNumber.get()]
            img_list[cropNumber.get()] = img
            l.configure(image=img)
            l.image = img
            cropNumber.set(cropNumber.get()+1)

# Remove last image from board
def removeImage():
    global imgLabel_list, pixel
    if (cropNumber.get() > 0):
        cropNumber.set(cropNumber.get()-1)
        imgLabel_list[cropNumber.get()].configure(image = pixel)
        imgLabel_list[cropNumber.get()].image = pixel
        img_list[cropNumber.get()] = pixel

        
def getFile():
    global imgLabel_list, pixel, videoLength
    # Getting file
    file = fd.askopenfilename()
    print('file selected: ' + file)
    currFile.set(file)
    frameNumber.set(0)
    cropNumber.set(0)
    
    # Updating frame
    img2 = getImage(frameNumber.get())
    videoFrame.configure(image=img2)
    videoFrame.image = img2

    # Updating video frame count
    cap = cv2.VideoCapture(file)
    videoLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    

    # Resetting board
    for i in range(10):
        imgLabel_list[i].configure(image = pixel)
        imgLabel_list[i].image = pixel

def makeTextLabel():
    global phaseNameList
    input = "   "
    spacing = ['     ', '     ', '    ', '    ']
    for i in range(len(phaseNameList)):
        if (i != len(phaseNameList)-1):
            input = input + phaseNameList[i] + spacing[i]
        else:
            input = input + phaseNameList[i]
    input = input.lower()
    input = input + "  "

    output = np.zeros((212, 145*len(input), 3))
    for i in range(len(input)):
        char = input[i]
        output[:, i*145:(i+1)*145] = getLetter(char)
    
    return output
    
        
    


# adds black pixels around photo to make sure everything fits properly when collated
def addPadToImageLabel(idx):
    global img_list
    image = np.array(ImageTk.getimage(img_list[idx]))

    img_w = image.shape[1]
    img_h = image.shape[0]

    width = int(GUI_width/5) 
    height = int(4.5*GUI_width/30)

    xStart = int(abs(width - img_w) / 2)
    yStart = int(abs(height - img_h) / 2)

    output = np.zeros((height, width, 4))

    output[yStart:yStart + img_h, xStart:xStart + img_w] = image[:,:]
    return output


def writeToFile():
    global imgLabel_list
    width = int(GUI_width/5) 
    height = int(4.5*GUI_width/30)
    output = np.zeros((int(2.2*height), int(5*width), 4))
    # Collating images from the board
    for i in range(2):
        for j in range(5):
            image = addPadToImageLabel(i*5 + j)
            if (i == 0):
                output[0:height, j*width:(j+1)*width] = image[:,:]
            else:
                output[-height-1: -1, j*width:(j+1)*width] = image[:,:]
                
    text_label = makeTextLabel()
    label = ResizeWithAspectRatio(text_label, width = output.shape[1])
    
    output_with_label = np.zeros((output.shape[0] + label.shape[0], output.shape[1], 3))
    output_with_label[:label.shape[0], :, :] = label[:,:,:]
    output_with_label[label.shape[0]:, :, :] = output[:,:,0:3]


    # Writing to a file
    converted_img = PIL.Image.fromarray(np.uint8(output_with_label[:,:,:])).convert('RGB')
    file = fd.asksaveasfilename(defaultextension=".jpg")
    converted_img.save(file)    
    converted_img.close()
    

# File Selector
fileSelect = Button(window, text = "Select File", width = 14, height = 2, command = getFile, bg = 'yellow')
fileSelect.place(x = int(5*GUI_width/8), y = int(GUI_height/22.5), anchor = 'n')

# Write to image file
saveBoard = Button(window, text = "Save Board", width = 20, height = 2, command = writeToFile, bg = 'yellow')
saveBoard.place(x = int(5*GUI_width/8), y = int(3*GUI_height/22.5), anchor = 'n')

# Increment/Decrement Frame buttons
incrButton = Button(window, text = "Next Frame", width = 14, height = 2, command = incrFrame, bg = 'lime')
incrButton.place(x = int(7*GUI_width/8), y = int(GUI_height/22.5), anchor = 'nw')
decrButton = Button(window, text = "Prev Frame", width = 14, height = 2, command = decrFrame, bg = 'orange')
decrButton.place(x = int(7*GUI_width/8), y = int(GUI_height/22.5), anchor = 'ne')

# Multi-Increment/Decrement Frame buttons
incrButton1 = Button(window, text = "Forward 10 Frames", width = 14, height = 2, command = incrFrame10, bg = 'lime')
incrButton1.place(x = int(7*GUI_width/8), y = int(2.5*GUI_height/22.5), anchor = 'nw')
decrButton1 = Button(window, text = "Back 10 Frames", width = 14, height = 2, command = decrFrame10, bg = 'orange')
decrButton1.place(x = int(7*GUI_width/8), y = int(2.5*GUI_height/22.5), anchor = 'ne')

# Add phase to board
addButton = Button(window, text = "Add Frame to Board", width = 29, height = 2, command = pasteImage, bg = 'cyan')
addButton.place(x = int(7*GUI_width/8), y = int(4.5*GUI_height/22.5), anchor = 'n')

# Delete image from board
delButton = Button(window, text = "Delete Frame from Board", width = 29, height = 2, command = removeImage, bg = 'cyan')
delButton.place(x = int(7*GUI_width/8), y = int(6*GUI_height/22.5), anchor = 'n')


window.mainloop()

