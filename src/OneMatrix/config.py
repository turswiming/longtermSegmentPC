from detectron2.config import CfgNode as CN

def add_OneMatrix_config(cfg):

    cfg.MODEL.ONEMATRIX = CN()
    cfg.MODEL.ONEMATRIX.N_CLASSES = 21
    cfg.MODEL.ONEMATRIX.BILINEAR = True
