from dataset import vipl_hr
from evaluate import metric, postprocess
from configs import running, preprocess
from model import loss_function, physformer


ops = vipl_hr.FramePreprocess(preprocess.FrameTrain)
ops.read_process()
