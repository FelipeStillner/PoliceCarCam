input_w = 32
input_h = 96
anchors = [[16, 58], [19, 67], [22, 75], [26, 84], [32, 96]]
masks = [[3, 4], [0, 1, 2]] 
OBJ_THRESH = 0.1
NMS_THRESH = 0.6

NUM_CLS = 26  # Number of classes for letters A-Z
MAX_BOXES = 500

# RKNN_MODEL_PATH = './neural_network/rknnModels/charRec.rknn'  # Path to  .rknn model
# RKNN_MODEL_PATH = './neural_network/rknnModels/quanti_charRec.rknn'  # Path to  .rknn model
RKNN_MODEL_PATH = './neural_network/rknnModels/kl_charRec.rknn'  # Path to  .rknn model
# RKNN_MODEL_PATH = './neural_network/rknnModels/mmse_charRec.rknn'  # Path to  .rknn model

# Class labels for letters A-Z
CLASSES = (
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
    )