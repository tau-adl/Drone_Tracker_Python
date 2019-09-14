# Interface
PREDICTION_ACTIVATE = 0

# Camera
FPS = 60  # Frames Per Second
DIFFERENTIAL_TIME = 1 / FPS  # time between two frames

# Tracking
N_TRK_CONF = 5  # minimum number of frames required for a track object to define as drone track
N_TRK_DEL = 8  # number of frames not detected to delete the track
N_TRK_DEL_FOR_NEW_TRACK = 3  # if object never been tracked more then this value, delete it

# RLS
SET_FOR_PREDICATION = 8
WEIGHT_OF_LAST_SAMPLE = 0.8
WLS_ALPHA = WEIGHT_OF_LAST_SAMPLE ** (1 / SET_FOR_PREDICATION)
GATING_RADIUS = 10

# time vector
NOT_VALID_VALUE = 5000
nfrm = 1

# groups
MAX_NUMBER_OF_TRACKS = 5  # don't track more than this number of objects

# main
REF_PATCH_SIZE_CROP = [100, 100]  # when detecting the drone, use this size around the middle pixel to move to the cnn
REF_PATCH_SIZE_CROP_2 = [200, 200]  # use this size around the middle pixel of the drone
SIGMA = 1
FAST_TH = 0.1  # threshold of the FAST detector
GROUPS_NUM = 20  # maximum number of objects in the drone small image
MINIMUM_TRACK_LIFE = 5  # minimum number of frames for a track to be valid

# Statistics prediction
# Interface
PREDICTION_MODEL_MODE = 0  # 1-x0,v0. 2-x0,v0,a0.

# time vector
TIME_GLOBAL = []
