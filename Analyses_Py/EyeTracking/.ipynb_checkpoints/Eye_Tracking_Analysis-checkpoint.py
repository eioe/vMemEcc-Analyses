#!/usr/bin/env python
# coding: utf-8

# # DeepLabCut for Pupil Tracking
# 
# To save a copy of this file to your Drive:
# **[File > Save a copy in Drive]**
# 
# To run this notebook, simply follow the instructions and and run the cells. You should already have a project folder with labeled data in your google drive.



import os
# os.environ["DLClight"]="True"   #This will surpress the GUI as it won't work on the cloud
# print(os.environ['CONDA_DEFAULT_ENV'])

#import deeplabcut
import deeplabcut


# In[6]:






#Setup project variables:

#EDIT THESE ACCORDING TO YOUR PROJECT DETAILS:
ProjectFolderName = '/u/fklotzsche/ptmp_link/test_dir'
DataFolderName = '/u/fklotzsche/ptmp_link/Experiments/vMemEcc/Data/SubjectData'
VideoType = '.mp4' 

#Don't edit the video path
#If you want to work with just a specific video, you can add the name to the 
#end of the path, otherwise the current path will allow all videos in the 
#folder to be analyzed
# videofile_path = ['/content/drive/My Drive/'+ProjectFolderName+'/videos/'] 
# videofile_path

#This creates a path variable that links to your google drive copy
#Do not edit this
path_config_file = ProjectFolderName+'/config.yaml'
path_config_file



# ## Analyzing Videos
# This function analyzes the videos.
# 
# The results are stored in a csv file in the same directory where the video is. 

# In[ ]:


# deeplabcut.analyze_videos(path_config_file,videofile_path, videotype=VideoType, save_as_csv=True)


# Use the following instead to run on data from single subjects in the "original" vMemEcc folder structure:
# 

# In[ ]:

for subid in ['VME_S20', 'VME_S21', 'VME_S22', 'VME_S23', 'VME_S24']:
  #  subid = 'VME_S27'
  path_sub = os.path.join(DataFolderName, subid, 'EyeTracking') 
  dirs_blocks = os.listdir(path_sub) 
  for dir in dirs_blocks:
    vids_path = [os.path.join(path_sub, dir, '000')]
    print("Running block " + dir)
    deeplabcut.analyze_videos(path_config_file,vids_path, videotype=VideoType, save_as_csv=True)
    #  deeplabcut.create_labeled_video(path_config_file, vids_path, videotype=VideoType, filtered=False, codec='jpeg')


# In[24]:



# ## Create Labeled Video
# This creates a video in .mp4 format with labels predicted by the network. This video is saved in the same directory where the original video resides. 

# In[ ]:


# deeplabcut.create_labeled_video(path_config_file, videofile_path, videotype=VideoType, filtered=False, codec='jpeg')



# for dir in dirs_blocks[2:]:
#   vids_path = [os.path.join(path_sub, dir, '000')]
#   print(vids_path)
#   deeplabcut.create_labeled_video(path_config_file, vids_path, videotype=VideoType, filtered=False, codec='jpeg')

