***Command***

* Train

python -u main.py --output_dir ./exps/coco2017/ms_detr_300 --with_box_refine --two_stage --dim_feedforward 2048 --epochs 12 --lr_drop 11 --coco_path=./data/coco2017 --num_queries 300 --use_ms_detr --use_aux_ffn --cls_loss_coef 1 --o2m_cls_loss_coef 2 --enc_cls_loss_coef 2 --enc_bbox_loss_coef 5 --enc_giou_loss_coef 2 > ./exps/coco2017/ms_detr_300/train.log


python -u main.py --output_dir ./exps/VOC_0712/ms_detr_300 --with_box_refine --two_stage --dim_feedforward 2048 --epochs 12 --lr_drop 11 --coco_path=./data/VOC_0712 --num_queries 300 --use_ms_detr --use_aux_ffn --cls_loss_coef 1 --o2m_cls_loss_coef 2 --enc_cls_loss_coef 2 --enc_bbox_loss_coef 5 --enc_giou_loss_coef 2  > ./exps/VOC_0712/ms_detr_300/train.log


* Eval

python -u main.py --output_dir ./exps/coco2017/ms_detr_300 --with_box_refine --two_stage --dim_feedforward 2048 --epochs 12 --lr_drop 11 --coco_path=./data/coco2017 --num_queries 300 --use_ms_detr --use_aux_ffn --topk_eval 100 --resume ./exps/coco2017/ms_detr_300/download_checkpoint.pth --eval > ./exps/coco2017/ms_detr_300/eval.log

python -u main.py --output_dir ./exps/coco2017/ms_detr_300_download --with_box_refine --two_stage --dim_feedforward 2048 --epochs 12 --lr_drop 11 --coco_path=./data/coco2017 --num_queries 300 --use_ms_detr --use_aux_ffn --topk_eval 100 --resume ./exps/coco2017/ms_detr_300_download/download_checkpoint.pth --eval > ./exps/coco2017/ms_detr_300_download/eval.log

python -u main.py --output_dir ./exps/VOC_0712/ms_detr_300 --with_box_refine --two_stage --dim_feedforward 2048 --epochs 12 --lr_drop 11 --coco_path=./data/VOC_0712 --num_queries 300 --use_ms_detr --use_aux_ffn --topk_eval 100 --resume ./exps/VOC_0712/ms_detr_300/checkpoint.pth --eval  > ./exps/VOC_0712/ms_detr_300/eval.log

python -u main.py --output_dir ./exps/OpenImages/ms_detr_300 --with_box_refine --two_stage --dim_feedforward 2048 --epochs 12 --lr_drop 11 --coco_path=./data/OpenImages --num_queries 300 --use_ms_detr --use_aux_ffn --topk_eval 100 --resume ./exps/OpenImages/ms_detr_300/COCO_pretrain.pth --eval > ./exps/OpenImages/ms_detr_300/eval.log

* Extract object specific features
python -u main.py --output_dir ./exps/coco2017/ms_detr_300_download --with_box_refine --two_stage --dim_feedforward 2048 --epochs 12 --lr_drop 11 --coco_path=./data/coco2017 --num_queries 300 --use_ms_detr --use_aux_ffn --topk_eval 100 --resume ./exps/coco2017/ms_detr_300_download/download_checkpoint.pth --eval --batch_size 1 --extract_ose

python -u main.py --output_dir ./exps/OpenImages/ms_detr_300 --with_box_refine --two_stage --dim_feedforward 2048 --epochs 12 --lr_drop 11 --coco_path=./data/OpenImages --num_queries 300 --use_ms_detr --use_aux_ffn --topk_eval 100 --resume ./exps/OpenImages/ms_detr_300/COCO_pretrain.pth --eval --batch_size 1 --extract_ose


***Object specific features***
myconfigs.py


***Dummy***
python -u main.py --output_dir ./exps/OpenImages/ms_detr_300 --with_box_refine --two_stage --dim_feedforward 2048 --epochs 12 --lr_drop 11 --coco_path=./data/OpenImages --num_queries 300 --use_ms_detr --use_aux_ffn --topk_eval 100 --resume ./exps/OpenImages/ms_detr_300/COCO_pretrain.pth



COCO after 1 epoch - n iterations == 59143, test = 2500 (batch)
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.109
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.165
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.114
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.061
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.136
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.159
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.218
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.333
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.401
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.175
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.427
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.586

COCO after 2 epoch
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.245
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.357
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.264
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.127
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.261
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.346
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.264
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.416
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.463
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.227
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.492
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.674

COCO after 3 epoch
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.291
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.417
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.314
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.152
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.320
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.395
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.283
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.455
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.499
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.259
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.543
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.695

COCO download weight
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.475
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.650
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.516
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.300
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.506
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.628
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.371
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.613
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.659
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.457
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.704
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.836



VOC0712 after 1 epoch - n_iteration == 8275, test = 2476 (batch), num classes 21


***Custom***
./models/deformable_detr.py: in the build function, set the num_classes=21


***Github***
https://github.com/Atten4Vis/MS-DETR


***Tmux***
export PATH=$PATH:/home/khoadv/tmux/usr/bin