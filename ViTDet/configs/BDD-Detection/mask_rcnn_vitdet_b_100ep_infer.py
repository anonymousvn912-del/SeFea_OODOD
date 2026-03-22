from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

from ..common.voc_loader import dataloader


model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model
model.roi_heads.num_classes=10
model.roi_heads.mask_in_features=None
model.roi_heads.box_predictor.test_score_thresh = 0.3794 ## Run eval_vitdet.py to get this value
# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = (
    "ViTDet/output/ViTDet/BDD-Detection/mask_rcnn_vitdet_b_100ep/model_final.pth"
    # "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"
)
train.output_dir="ViTDet/output/ViTDet/VOC-Detection/mask_rcnn_vitdet_b_100ep/infer_final"
train.eval_period=2500

dataloader.evaluator.output_dir = train.output_dir ## output_dir must be provided to COCOEvaluator for datasets not in COCO format.
dataloader.train.total_batch_size=4

# Schedule
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
train.max_iter = 50000

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[40000, 45000],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
