import torchvision


class TrainConfig:
    Fs = 30
    T = 160
    trans = torchvision.transforms.RandomHorizontalFlip()
    record = "./train_record.csv"
    folds = [2, 3, 4, 5]

    batch_size = 8
    # image_size = 128
    tau = 2
    num_epochs = 25

    device = "cuda:4"
    device_ids = [4, 5, 6, 7]


class TestConfig:
    Fs = 30
    T = 300
    trans = None
    record = "./test_record.csv"
    folds = [1]

    batch_size = 1
    # image_size = 128
    tau = 2
    methods = ["MAE", "RMSE", "MAPE", "R"]
    post = "fft"

    device = "cuda:4"
