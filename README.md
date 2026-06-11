# Facial Emotion Recognition CNN
This is my first model that would tell the emotions based on a live feed from your computer 

# Datasets
We are using the FER 2013 dataset from kaggle you can install it from kaggle. Even tho it has been said to be very notoriously cluttered and not very clear. I have bee able to get about 60% accuracy (For context the best so far in this dataset is about 78%0)

The link below is the dataset link in kaggle:
https://www.kaggle.com/datasets/msambare/fer2013?select=test

# Resources
In order to save huge amounds of time i have used this pre trained model the google the "MobileNetV2" file. This is basically a pre trained neural network that was trained on super computers for multiple days by google for this exact purpose!

Also the aarcascade_frontalface_default.xml file you see in the gitignore file is the ML modle that was trained specifically to idnetify face from the whole wide camera and trained by OpenCV. It has been imported from this git hub link given below:
https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml


I had to switch from VS code to Colab for training the model on the T4 GPU. My intel core mac wasnt able to handle all the epoches at all hence i used the Gooogle Colab on this and ran on the cloud.


# AI Declaration 
1. So far to find out about this data i used perplexity for research purposes.
   
2. To be honest since this project is my official shift from building basic machine learning models to this Convulation Nural Network one i used AI for some codes in a few sections of this ipynb files and then used CLAUDE for debugging purposes of this as well.


## Note

We provide two deployment versions because WebRTC facilitates low-latency, continuous streaming (high complexity/fragile dependencies), whereas camera_input ensures cross-platform platform stability (native browser-processing/zero system dependencies) for guaranteed uptime during live demos.

If you have any questions regarding this and how i built this i have the whole 'learning logs' folder for this purpose with .txt files acting as like a guide for you.

## DEMO
The demo for the live tracking is the local server:

https://github.com/user-attachments/assets/59a4aab1-9bc0-472b-b3f3-395e3d593117


This is the photo only mode demo in the live website:
<img width="1280" height="800" alt="Screen Shot 2026-06-11 at 23 23 25" src="https://github.com/user-attachments/assets/aa6cc016-fdcf-4618-9277-94e64043386b" />

<img width="1280" height="800" alt="Screen Shot 2026-06-11 at 23 29 31" src="https://github.com/user-attachments/assets/ad02aebc-de8c-4375-a528-7265110b1179" />

<img width="1280" height="800" alt="Screen Shot 2026-06-11 at 23 28 37" src="https://github.com/user-attachments/assets/deb33c4c-6364-4968-af43-81ec02b8170d" />

<img width="1280" height="800" alt="Screen Shot 2026-06-11 at 23 27 56" src="https://github.com/user-attachments/assets/354f55b2-f419-4459-b4b5-9924cf7a4254" />

<img width="1280" height="800" alt="Screen Shot 2026-06-11 at 23 27 34" src="https://github.com/user-attachments/assets/67e5ffae-772e-40e4-9788-737386d34a83" />



   
