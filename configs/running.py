import torchvision


class TrainConfig:
    # T = 300
    T = 160
    # trans = torchvision.transforms.RandomHorizontalFlip()
    trans = None
    record = "./160_record.csv"
    # record = "./train_record.csv"
    folds = [2, 3, 4, 5]
    tasks = []
    sources = [1, 2, 3]

    batch_size = 4
    # image_size = 128
    tau = 2
    num_epochs = 9  # it seems 25 epochs will cause overfitting

    device = "cuda:1"  # cuda:0
    device_ids = [4, 5, 6, 7]


class TestConfig:
    T = 160  # 160
    trans = None
    record = "./160_record.csv"
    folds = [1]
    tasks = []
    sources = [1, 2, 3]

    batch_size = 1
    # image_size = 128
    tau = 2
    methods = ["MAE", "RMSE", "MAPE", "R"]
    post = "fft"
    diff = False
    detrend = True

    device = "cuda:1"  # cpu
