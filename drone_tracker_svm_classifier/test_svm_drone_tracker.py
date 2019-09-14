import os
from sklearn.model_selection import train_test_split
import time
import cv2
from joblib import load


drones_directory = r"C:\Users\roiei\PycharmProjects\opencv_alexnet\drones_db\Raw_Data_Drones"
drone_images_paths = list()
for (dirpath, dirnames, filenames) in os.walk(drones_directory):
    drone_images_paths_to_add = [os.path.join(dirpath, file) for file in filenames]
    drone_images_paths += drone_images_paths_to_add

backgrounds_directory = r"C:\Users\roiei\PycharmProjects\opencv_alexnet\drones_db\Raw_Data_Background"
background_images_paths = list()
for (dirpath, dirnames, filenames) in os.walk(backgrounds_directory):
    background_images_paths_to_add = [os.path.join(dirpath, file) for file in filenames]
    background_images_paths += background_images_paths_to_add

image_paths_list = drone_images_paths + background_images_paths
image_labels_list = ["drone"]*len(drone_images_paths) + ["background"]*len(background_images_paths)

image_paths_train, image_paths_test, labels_train, labels_test = train_test_split(
    image_paths_list, image_labels_list, test_size=0.4, stratify=image_labels_list, random_state=42)

print("reading test images...")
start = time.time()
images_test = []
for p in image_paths_test:
    image = cv2.imread(p)
    image = cv2.resize(image, (224, 224))
    images_test.append(image)
# convert the images list into an OpenCV-compatible blob
blob_test = cv2.dnn.blobFromImages(images_test, 1, (224, 224), (104, 117, 123))
end = time.time()
print("reading test images and converting to blob took {:.5} seconds".format(end - start))

prototxt_alexnet_fc7 = "C:/Users/roiei/Desktop/caffe/caffe/models/bvlc_alexnet/deploy_fc7.prototxt"
model_alexnet = "C:/Users/roiei/Desktop/caffe/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel"

# load our serialized model from disk
print("loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt_alexnet_fc7, model_alexnet)

print("test data forward propagation...")
# set the blob as input to the network and perform a forward-pass to
# obtain our output classification
net.setInput(blob_test)
start = time.time()
test_fc7_activations = net.forward()
end = time.time()
print("forward propagation to get test data fc7 activations took {:.5} seconds".format(end - start))

print("loading SVM classifier...")
start = time.time()
clf = load('alexnet_svm.joblib')
end = time.time()
print("loading SVM classifier took {:.5} seconds".format(end - start))

print("calculating score over test data...")
start = time.time()
test_score = clf.score(test_fc7_activations, labels_test)
end = time.time()
print('score over test data is: ', test_score)
print("calculating score over test data took {:.5} seconds".format(end - start))
