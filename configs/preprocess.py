class FrameTrain:
    """for RhythmNet, dynamic detection and no large"""
    input_path = "D:\\papers\\Video\\vipl_test"
    record_path = "./train_record.csv"

    MODIFY = False
    W = 128
    H = 128
    DYNAMIC_DETECTION = True
    DYNAMIC_DETECTION_FREQUENCY = 1
    CROP_FACE = True
    LARGE_FACE_BOX = False
    LARGE_BOX_COEF = 1.

    DO_CHUNK = True
    CHUNK_LENGTH = 160
    CHUNK_STRIDE = -1


class FrameTest:
    input_path = "D:\\papers\\Video\\vipl_test"
    record_path = "./test_record.csv"

    MODIFY = False
    W = 128
    H = 128
    DYNAMIC_DETECTION = True
    DYNAMIC_DETECTION_FREQUENCY = 1
    CROP_FACE = True
    LARGE_FACE_BOX = False
    LARGE_BOX_COEF = 1.

    DO_CHUNK = True
    CHUNK_LENGTH = 300
    CHUNK_STRIDE = -1
