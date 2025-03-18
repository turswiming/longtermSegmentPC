from detectron2.config import CfgNode as CN

def add_unet_config(cfg):
    cfg.MODEL.UNET = CN()
    cfg.MODEL.UNET.N_CHANNELS = 3
    cfg.MODEL.UNET.N_CLASSES = 21
    cfg.MODEL.UNET.BILINEAR = True