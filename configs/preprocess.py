class Frame160:
    input_path = ""
    record_path = "./160_record.csv"
    img_cache = ""
    gt_cache = ""

    MODIFY = False
    W = 128
    H = 128
    DYNAMIC_DETECTION = False
    DYNAMIC_DETECTION_FREQUENCY = -1
    CROP_FACE = True
    LARGE_FACE_BOX = False
    LARGE_BOX_COEF = 1.0

    DO_CHUNK = True
    CHUNK_LENGTH = 160
    CHUNK_STRIDE = -1


class Frame300:
    input_path = ""
    record_path = "./300_record.csv"
    img_cache = ""
    gt_cache = ""

    MODIFY = False
    W = 128
    H = 128
    DYNAMIC_DETECTION = False
    DYNAMIC_DETECTION_FREQUENCY = -1
    CROP_FACE = True
    LARGE_FACE_BOX = False
    LARGE_BOX_COEF = 1.0

    DO_CHUNK = True
    CHUNK_LENGTH = 300
    CHUNK_STRIDE = -1
