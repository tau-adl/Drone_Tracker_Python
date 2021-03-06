import os
from sklearn.model_selection import train_test_split
import time
import cv2
from sklearn import svm
from joblib import dump

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

print("reading train images...")
start = time.time()
images_train = []
for p in image_paths_train:
    image = cv2.imread(p)
    image = cv2.resize(image, (224, 224))
    images_train.append(image)

# convert the images list into an OpenCV-compatible blob
blob_train = cv2.dnn.blobFromImages(images_train, 1, (224, 224), (104, 117, 123))
end = time.time()
print("reading train images and converting to blob took {:.5} seconds".format(end - start))

prototxt_alexnet_fc7 = "C:/Users/roiei/Desktop/caffe/caffe/models/bvlc_alexnet/deploy_fc7.prototxt"
model_alexnet = "C:/Users/roiei/Desktop/caffe/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel"

# load our serialized model from disk
print("loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt_alexnet_fc7, model_alexnet)

print("train data forward propagation...")
# set the blob as input to the network and perform a forward-pass to
# obtain our output classification
net.setInput(blob_train)
start = time.time()
fc7_activations = net.forward()
end = time.time()
print("forward propagation to get train data fc7 activations took {:.5} seconds".format(end - start))

print("training SVM...")
start = time.time()
clf = svm.SVC(gamma='scale', kernel='rbf', probability=True)
clf.fit(fc7_activations, labels_train)
end = time.time()
print("training SVM took {:.5} seconds".format(end - start))
print("saving trained SVM...")
dump(clf, 'alexnet_svm.joblib')




