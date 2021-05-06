# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
from os import path as op
import numpy as np
import pandas as pd
import cv2
import skimage.measure as meas


# %%
# save images with ellipse?
save_imgs = True

# display images in pop up window?
# NB: to go to the next image, click on the image and press any button.
#     You have to o through all images that are opened one after the other. 
#     If you close the pop up window, the script crashes. 
#     Choose the step size wisely. Or for looking at many images, I'd recommend
#     to save them (w/o displaying) and look at them in the folder.
show_imgs = False

# Additionally plot the tracked points. Normally not necessary. 
# Only for checking whether everything works correctly. 
plot_points = False

# Path to "videos" folder:
path_main = op.join('C:\\', 'Users', 'Felix', 'Downloads', 'videos')

# Path to save images to:
path_out = op.join(path_main, 'ell_out')
if not op.exists(path_out):
    os.makedirs(path_out)
    print('creating dir: ' + path_out) 


subID = 'VME_S01'
blockNR = '13'


# %%
# get tagged data from the csv:
fname = op.join(path_main, '_Block'.join([subID, blockNR]) + 'DLC_resnet50_testNov6shuffle1_550000.csv')
dlc_df = pd.read_csv(fname, skiprows=2)


# %%
ell_ = meas.EllipseModel()

def fun_(data): 
    data = np.reshape(data.to_numpy(), (-1,2)) #need to reshape each row into (Nx2) shapefor fitting the ellipse
    if np.isnan(np.sum(data)): # for NA values we write out zeros
        outp = [0] * 5
        return outp
    else:
        ell_.estimate(data)
        return ell_.params


# %%
# rowwise fit ellipse (this takes a while; ~30s on my machine)
dlc_df['ell_x'], dlc_df['ell_y'], dlc_df['ell_a'], dlc_df['ell_b'], dlc_df['ell_theta'] = zip(*dlc_df[['x', 'y', 
                                                                                                       'x.1', 'y.1', 
                                                                                                       'x.2', 'y.2', 
                                                                                                       'x.3', 'y.3', 
                                                                                                       'x.4', 'y.4', 
                                                                                                       'x.5', 'y.5']].apply(fun_, axis=1))



# %%
# get video:
fname = op.join(path_main, '_Block'.join([subID, blockNR])+'DLC_resnet50_testNov6shuffle1_550000_filtered_labeled.mp4')
cap = cv2.VideoCapture(fname)


# %%
start = 36000-2000
end = 37000 # dlc_df.shape[0]
step = 100
for i in np.arange(start, end, step): #step through frames
    obs = dlc_df.iloc[i,:] #grab a single row from the df
    cap.set(1, i) # scroll to according frame in video
    rval, frame = cap.read() #grab frame
    cv2.ellipse(img=frame, 
                center=(int(obs['ell_x']), int(obs['ell_y'])), 
                axes=(int(obs['ell_a']), int(obs['ell_b'])),
                angle=np.rad2deg(obs['ell_theta']), 
                startAngle=0, 
                endAngle=360, 
                color=(1,1,254))
    if plot_points:
        for j in range(6):
            if j==0: 
                add_str = ''
            else:
                add_str = '.'+str(j)
            cv2.circle(
                img=frame,
                center=(int(obs['x'+add_str]), int(obs['y'+add_str])),
                radius=1,
                color=(250, 250, 123), 
                thickness=-1)
    if save_imgs:
        cv2.imwrite(op.join(path_out, 'frame'+str(i)+'.jpg'), frame)
    if show_imgs:
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 600, 600)
        cv2.imshow('image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# %%
