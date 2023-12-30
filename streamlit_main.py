import cv2
from ultralytics import YOLO
import numpy as np
import PIL
from PIL import Image  
import pandas as pd
import streamlit as st
from io import BytesIO
import os

st.set_page_config(layout="wide")
st.header('Cropping App')
st.subheader('Please allow the program to run for a few seconds after pressing a button')
st.subheader('This mainly applies to the \"add to board\" button')

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

def make_ss(name, init):
    if name not in st.session_state:
        st.session_state[name] = init

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
    current_path = os.path.abspath(__file__)
    curr_dir = current_path[:current_path.rfind("\\")+1]
    file_path = curr_dir + 'letters.jpg'
    
    letters = cv2.imread(file_path)

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

# Gets video from mp4 file
cap = cv2.VideoCapture('black_screen.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
success, img = cap.read()


file = st.file_uploader("Select File", type = ["mp4", "mov", "avi"], 
                         accept_multiple_files = False, label_visibility = "hidden")

def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())

#make_ss(name = 'total_frame_count', init = 0)
if 'total_frame_count' not in st.session_state:
    st.session_state['total_frame_count'] = 0

# Copies selected file to local file
temp_file_to_save = './temp_file_1.mp4'

video_panel = st.container()

curr_frame = None
if file:
    write_bytesio_to_file(temp_file_to_save, file)

    cap = cv2.VideoCapture(temp_file_to_save)
    st.session_state.total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state['frame'])
    success, img = cap.read()
    img = ResizeWithAspectRatio(img, width = 1000, height = 600)
    curr_frame = img
    if (img is None):
        st.text("NONE")
    img = PIL.Image.fromarray(np.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    video_panel.image(img)



#Frame Counter
#make_ss(name = 'frame', init = 1)
if 'frame' not in st.session_state:  
    st.session_state['frame'] = 1

def incrFrame():
    if st.session_state.frame < st.session_state.total_frame_count - 1:
        st.session_state['frame'] = st.session_state['frame'] + 1
    st.text(str(st.session_state['frame']))

def decrFrame():
    if st.session_state.frame > 0:
        st.session_state['frame'] = st.session_state['frame'] - 1
        st.text(str(st.session_state['frame']))

def incrFrame10():
    if st.session_state.frame < st.session_state.total_frame_count - 10:
        st.session_state['frame'] = st.session_state['frame'] + 10
    st.text(str(st.session_state['frame']))

def decrFrame10():
    if st.session_state.frame > 10:
        st.session_state['frame'] = st.session_state['frame'] - 10
        st.text(str(st.session_state['frame']))

# Number of crops
#make_ss('num_crops', 0)
if 'num_crops' not in st.session_state:
    st.session_state['num_crops'] = 0

# Bounding Box Variables
if 'box_x1' not in st.session_state:
    st.session_state['box_x1'] = -1
if 'box_x2' not in st.session_state:
    st.session_state['box_x2'] = -1
if 'box_y1' not in st.session_state:
    st.session_state['box_y1'] = -1
if 'box_y1' not in st.session_state:
    st.session_state['box_y2'] = -1

# Extrapolation vars

if 'cropped_frame_numbers' not in st.session_state:
    st.session_state['cropped_frame_numbers'] = pd.DataFrame()

if 'cropped_frame_x' not in st.session_state:
    st.session_state['cropped_frame_x'] = pd.DataFrame()
 

if 'crop_num' not in st.session_state:
    st.session_state['crop_num'] = 0

# YOLO model
model = YOLO('yolov8m.pt')
results = []
results = model(img, conf = 0.3)


# tries to extrapolate the x coords if the detector fails
if (st.session_state.cropped_frame_x.size >= 2):
    box_w = abs(st.session_state.box_x1 - st.session_state.box_x2)
    st.session_state.box_x1.set(extrapolate_box_x1(st.session_state.cropped_frame_numbers[-1], st.session_state.cropped_frame_numbers[-2], 
                                  st.session_state.frame_number.get(), st.session_state.cropped_frame_x[-1], st.session_state.cropped_frame_x[-2]))
    st.session_state.box_x2 = st.session_state.box_x1 + box_w

# Sets box tensor based on detector
if (len(results) > 0):
    boxes = results[0].boxes
    if(len(boxes) > 0):
        box_tensor = boxes[0].xyxy
        st.session_state.box_x1 = int(box_tensor[0][0].item())
        st.session_state.box_y1 = int(box_tensor[0][1].item())
        st.session_state.box_x2 = int(box_tensor[0][2].item())
        st.session_state.box_y2 = int(box_tensor[0][3].item())

# Gets nth frame (tkinter img)
def getImage(n):
    global img_width, img_height
    cap = cv2.VideoCapture(temp_file_to_save)
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame)
    success, img = cap.read()
    
    img = ResizeWithAspectRatio(img, img_width, img_height)
    return img

# Prevent errors when cropping
def adjustXBounds(n):
    global img_width
    return int(min(img_width, max(0, n)))

def adjustYBounds(n):
    global img_height
    return int(min(img_height, max(0, n)))

# Crops image based on Pose Detection
def cropImage():
    global img_width, img_height, cropped_frame_numbers, cropped_frame_x

    # adding frame number of crop for x coordinate extrapolation
    st.session_state.cropped_frame_numbers.add(st.session_state.frame)
    cap = cv2.VideoCapture(temp_file_to_save)
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame)
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
            st.session_state.box_x1 = int(box_tensor[0][0].item())
            st.session_state.box_y1 = int(box_tensor[0][1].item())
            st.session_state.box_x2 = int(box_tensor[0][2].item())
            st.session_state.box_y2 = int(box_tensor[0][3].item())

    # adds padding to detector box + adds a new x coordinate reference for position extrapolation
    x1, y1, x2, y2 = GetBoundingBox(st.session_state.box_x1, st.session_state.box_y1, st.session_state.box_x2, 
                                    st.session_state.box_y2, img_width, img_height)
    st.session_state.cropped_frame_x.add(x1)
    
    # if the detection is successful then return the cropped image
    if (x1 >= 0 and y1 >= 0):
        crop_img = img[y1:y2, x1:x2]
        image1 = PIL.Image.fromarray(np.uint8(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)))
        ratio = float(abs(x1-x2))/float(abs(y1-y2))

        image2 = image1.resize((int(ratio*4.5*GUI_width/30), int(4.5*GUI_width/30)))
        return (0, image2)
    
    # if detection is not successful then return a fail code and a pixel image
    listImg = np.array([[0]])
    img = PIL.Image.fromarray(cv2.cvtColor(listImg, cv2.COLOR_BGR2RGB))
    return (1, img)

button_container = st.container()
col1, col2, col3, col4 = button_container.columns(4)
decr_frame = col1.button('Prev Frame', on_click = decrFrame)
incr_frame = col2.button('Next Frame', on_click = incrFrame)
decr10_frame = col3.button('Back 10 Frames', on_click = decrFrame10)
incr10_frame = col4.button('Forward 10 Frames', on_click = incrFrame10)

st.write('###')


if 'been_cropped' not in st.session_state:
    st.session_state['been_cropped'] = pd.DataFrame(np.zeros(10))
    

temp_images = []
for i in range(10):
    file_name = './temp_file_' + str(i) + '.jpg'
    temp_images.append(file_name)


pixel = PIL.Image.fromarray(np.uint8(np.array([[[0, 0, 0]]])))
image_bytes_io = BytesIO()
pixel.save(image_bytes_io, format = "JPEG")

for i in range(10):    
    if (st.session_state.been_cropped.at[i,0] == 0):
        write_bytesio_to_file(temp_images[i], image_bytes_io)



# Add an image to the board
def pasteImage(): 
    if(st.session_state.crop_num < 10):
        success, img = cropImage()
        image_bytes_io = BytesIO()
        img.save(image_bytes_io, format = "JPEG")

        if (success == 0):
            write_bytesio_to_file(temp_images[st.session_state.crop_num], image_bytes_io)
            st.session_state.been_cropped.at[st.session_state.crop_num, 0] = 1
            st.session_state.crop_num = st.session_state.crop_num + 1

def removeImage():
    if (st.session_state.crop_num > 0):
        st.session_state.crop_num = st.session_state.crop_num - 1
        st.session_state.been_cropped.at[st.session_state.crop_num, 0] = 0

add_remove = st.container()
col1, col2 = add_remove.columns(2)
add_frame = col1.button('Add Image to Board', on_click = pasteImage)
remove_frame = col2.button('Remove Image from Board', on_click = removeImage)

st.write('###')
st.write('###')

text_labels = st.container()
col1, col2, col3, col4, col5 = text_labels.columns(5)
col1.text('Toe off')
col2.text('Max Vertical Pos')
col3.text('Strike')
col4.text('Touch Down')
col5.text('Full Support')


fst_row = st.container()
col11, col12, col13, col14, col15 = fst_row.columns(5)

snd_row = st.container()
col21, col22, col23, col24, col25 = snd_row.columns(5)

images = [col11, col12, col13, col14, col15, col21, col22, col23, col24, col25]

for i in range(10):
    pil_img = Image.open(temp_images[i])
    images[i].image(pil_img)

st.write('###')
st.write('###')


# Makes the text label 
def makeTextLabel():
    phaseNameList = ['Toe off', 'Max Vertical Pos', 'Strike', 'Touch Down', 'Full Support']
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
    image = np.array(Image.open(temp_images[idx]))

    img_w = image.shape[1]
    img_h = image.shape[0]

    width = int(GUI_width/5) 
    height = int(4.5*GUI_width/30)

    xStart = int(abs(width - img_w) / 2)
    yStart = int(abs(height - img_h) / 2)

    output = np.zeros((height, width, 3))

    output[yStart:yStart + img_h, xStart:xStart + img_w] = image[:,:,0:3]

    # output shape (:, :, 3)
    return output

def getExportableImage():
    global imgLabel_list
    width = int(GUI_width/5) 
    height = int(4.5*GUI_width/30)
    output = np.zeros((int(2.2*height), int(5*width), 3))
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

    return output_with_label

st.write('###')
st.write('###')
st.write('###')

# Writing to a file
pil_img = PIL.Image.fromarray(np.uint8(getExportableImage()))
image_bytes = BytesIO()
pil_img.save(image_bytes, format = 'JPEG')

st.download_button(
    label = 'Download Board',
    data = image_bytes.getvalue(), 
    file_name = "board.jpg",
    mime = "image/jpg"
)
