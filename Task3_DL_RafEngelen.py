# %% [markdown]
# # Report task 3: Compare my model to Google's model
# 
# ## Introduction
# 
# The purpose of this report is to compare a Deep Learning model to a model created by the Teachable Machine by Google
# 
# In this file you will find some code written in comments. This is code that is only necessary to run on Streamlit. The version on Streamlit will have that code not in comments. 
# 
# Streamlit link:  

# %% [markdown]
# ## Scraping for images
# 
# First of all, I need images so that I can train our Deep Learning model. This needs a lot of images (100+) and the more images gathered, the better the model is trained. 

# %%

import os


# %% [markdown]
# Next we need our different categories we want to train the model for. I have chosen to characters in Star Wars. So the goal of our algorithm is to tell which character there is on an image. For structure, i create a folders. The main "images" folder and there are folders for each category.

# %%
categories=["Darth Vader", "Yoda", "Luke Skywalker", "R2D2", "C3PO"]
data_dir="./images/"
def setup_directories(categories):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for category in categories:
        directory_path = data_dir+category.replace(" ", "_")
        print(directory_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
    

#setup_directories(categories)

# %% [markdown]
# I have chosen not to create a regular web scraper to get images. I use fastbook, with this module i can connect to some API's, the API i am sending requests to is the DuckDuckGo images API. After some testing I have noticed that with regular web scrapers, the program quits after a set amount of images. For Google this was 20 and for Bing this was 35. If i wanted to gather 100+ images per category, i would have to make multiple web scrapers, so this seemed as the better option.
# 
# First i loop through the categories to gain 150 results. I have chosen 150 images, because I have noticed that not all the images can be downloaded this way. That's why I've chosen a number bigger than 100 and after executing the program I also have more than 100 images left.
# 
# After that I loop through all the image links and I send a get request to get the image. I check for errors and also for if the response status is 200. I save the image in the correct folder with a unique name. I also print errors in case they happen.

# %%
category_names = []


#get_images(categories)

# %% [markdown]
# Now the dataset for our program is complete.

# %% [markdown]
# ## EDA & Prep the data
# 
# ### EDA
# 
# Here I show how many images there are for each category. After that I use MatPlotLb to show some of the images.

# %%
import matplotlib.pyplot as plt

# %% [markdown]
# Now I want to validate that all images can be loaded. During the creation of this program, I have gotten some errors with training the model. With this extra check the errors are gone.

# %%
def check_image(img_path):
    
    try:
        img = Image.open(img_path)
        img.load()

        if img.mode != "RGBA":
            img = img.convert("RGBA")
        img.close()
        
    except (IOError, OSError) as e:
        print(f"Error loading image at path: {img_path}")
        if os.path.exists(img_path):
            os.remove(img_path)



    

# %% [markdown]
# I count the amount of images that are left and display a few of them.

# %%
def count_images(category_names):
    lowest_total = 100000
    directories= []
    for category in category_names:
        category_path = os.path.join(data_dir, category)
        num_images = len(os.listdir(category_path))
        print(f"Category: {category}, Number of Images: {num_images}")

        lowest_total = min(lowest_total, num_images)
        directories.append(category_path)

    display_images(category_names)

def display_images(category_names, amount_category_images=2):
    for category in category_names:

        #/{category_name}", category.replace(" ", "")
        category_path = os.path.join(data_dir, category)
        image_files = os.listdir(category_path)[:amount_category_images]  # Displaying the first 5 images
        for image_file in image_files:
            image_path = os.path.join(category_path, image_file)
            image = Image.open(image_path)
            plt.imshow(image)
            plt.title(category)
            plt.show()
    
# count_images(category_names)

# %% [markdown]
# ### Prep the data
# 
# 

# %% [markdown]
# The images are in directories according to their category, not according to training and testing dataset. We can use DataFrames with the ImageDataGenerator (ImageDataGenerator, z.d.).This will however give us some problems with Teachable Machine. We don't know which images are going in which dataset so we will split the images according to directories. 

# %%
import os
import shutil
import random

def split_images(train_dir="./images/train", test_dir="./images/test", split_ratio=0.9):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    # Iterate through each class directory
    for category in categories:
        category=category.replace(" ", "_")
        category_dir = os.path.join(data_dir, category)
        
        # Create train and test directories for each class
        train_class_dir = os.path.join(train_dir, category)
        os.makedirs(train_class_dir, exist_ok=True)

        test_class_dir = os.path.join(test_dir, category)
        os.makedirs(test_class_dir, exist_ok=True)

        # List all files in the class directory
        files = os.listdir(category_dir)

        # Calculate the number of files for training
        num_train_files = int(len(files) * split_ratio)

        # Randomly shuffle the list of files
        random.shuffle(files)

        # Move files to train directory
        for file in files[:num_train_files]:
            src_path = os.path.join(category_dir, file)
            dst_path = os.path.join(train_class_dir, file)
            shutil.move(src_path, dst_path)

        # Move remaining files to test directory
        for file in files[num_train_files:]:
            src_path = os.path.join(category_dir, file)
            dst_path = os.path.join(test_class_dir, file)
            shutil.move(src_path, dst_path)




#split_images()

# %% [markdown]
# I create datasets based on the different directories.

# %%
from keras.preprocessing.image import ImageDataGenerator


def create_data_sets():

    

    test_datagen = ImageDataGenerator(rescale = 1./255)
    # Set the parameters for your data

    
    test_dir='./images/test'
   
# Create the testing dataset from the 'test' directory
    test_ds = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size = (64, 64),
        batch_size = 32,
        class_mode = 'categorical'
    )
    return test_ds
test_set = create_data_sets()

# %% [markdown]
# ## Design a CNN

# %%





# %% [markdown]
# ## Train model

# %%



# %%
from keras.models import load_model 

model = load_model("models/modelStarWars.tf")

# %% [markdown]
# ## Comparison

# %% [markdown]
# ### Teachable Machine
# 
# First I will load in the Teachable Machine model. I have created this using the same training set. The code is provided on the Teachable Machine website when exporting the model.

# %%
 # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

def setup_tm():
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    tm_model = load_model("./models/teachable_machine/keras_model.h5", compile=False)

    tm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return tm_model
tm_model = setup_tm()



# %% [markdown]
# Here you can see 2 figures of the loss and the accuracy of the history of the model.

# %%
# Create a figure and a grid of subplots with a single call


# %% [markdown]
# ### Testing
# 
# Now I will use the test dataset to test both the Teachable Machine and my own model.

# %%
import pandas as pd

test_loss, test_acc = model.evaluate(test_set)
print('My model, test accuracy :', test_acc)
print('My model, test loss :', test_loss)

test_loss_tm, test_acc_tm = tm_model.evaluate(test_set)
print('Teachable Machine, test accuracy:', test_acc_tm)
print('Teachable Machine, test loss:', test_loss_tm)


data = {
    'Accuracy': [test_acc, test_acc_tm],
    'Loss': [test_loss, test_loss_tm]
}

df = pd.DataFrame(data, index=['My Model', 'Teachable Machine'])
df

# %% [markdown]
# As you can see, my model scores has a higher accuracy than the Teachable Machine by Google. I am perplexed by this, because i don't think the numbers are accurate. 25% accuracy is very low, especially for the Teachable Machine which is a good Deep Learning Algorithm.

# %% [markdown]
# ## Streamlit

# %% [markdown]
# This is what i show on Streamlit. On streamlit, the program will not scrape images and create a new model. The model that we have saved in this document will be added to streamlit.

# %%
import streamlit as st
st.header('Raf Engelen - r0901812 - 3APP01', divider='gray')
st.title("Task 3 ML: Comparing Deep Learning model to Teachable Machine")

st.write(df)


# %% [markdown]
# ## Literatuurlijst
# 
# ImageDataGenerator. (z.d.). tensorflow. https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_dataframe
# 
# Teachable machine. (z.d.). https://teachablemachine.withgoogle.com/


