import gradio as gr
from PIL import Image, ImageDraw
import os
import json
from MS_DETR import main

short_category_name = {'person': 'pe', 'bicycle': 'bi_cy', 'car': 'car', 'motorcycle': 'mo_to', 'airplane': 'ai','bus': 'bus','train': 'train','truck': 'truck','boat': 'boat','traffic light': 't_li','fire hydrant': 'f_hy','stop sign': 's_sig','parking meter': 'pa_me','bench': 'bench','bird': 'bird','cat': 'cat','dog': 'dog','horse': 'horse','sheep': 'sheep','cow': 'cow','elephant': 'el','bear': 'bear','zebra': 'ze','giraffe': 'gi_fe','backpack': 'ba_pa','umbrella': 'um','handbag': 'ha','tie': 'tie','suitcase': 'su_ca','frisbee': 'fr','skis': 'skis','snowboard': 'sn_bo','sports ball': 'sp_ba','kite': 'kite','baseball bat': 'ba_ba','baseball glove': 'ba_gl','skateboard': 'sk_bo','surfboard': 'su_bo','tennis racket': 'te_ra','bottle': 'bo','wine glass': 'wi_gl','cup': 'cup','fork': 'fork','knife': 'knife','spoon': 'sp','bowl': 'bowl','banana': 'ba','apple': 'apple','sandwich': 'sa','orange': 'or','broccoli': 'br_co','carrot': 'crt','hot dog': 'ho_do','pizza': 'pi','donut': 'donut','cake': 'cake','chair': 'chair','couch': 'couch','potted plant': 'po_pl','bed': 'bed','dining table': 'di_ta','toilet': 'to_le','tv': 'tv','laptop': 'la','mouse': 'mouse','remote': 'remote','keyboard': 'ke_b','cell phone': 'ce_ph','microwave': 'mi_wa','oven': 'oven','toaster': 'toaster','sink': 'sink','refrigerator': 're_fr','book': 'book','clock': 'cl','vase': 'va','scissors': 'sc','teddy bear': 'te_be','hair drier': 'ha_dr','toothbrush': 'toothbrush','aeroplane': 'ae','diningtable': 'dita','motorbike': 'motorbike','pottedplant': 'popl','sofa': 'sofa','tvmonitor': 'tv_mo','pedestrian': 'pedestrian','rider': 'rider','traffic sign': 't_si'}
voc_id_name = ["airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat","chair", "cow", "dining table", "dog", "horse", "motorcycle", "person","potted plant", "sheep", "sofa", "train", "tv"]
voc_short_category_name = {i: short_category_name[i] for i in voc_id_name}
bdd_id_name = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"]
bdd_short_category_name = {i: short_category_name[i] for i in bdd_id_name}

# ==== Config ====
def load_and_draw_boxes(image: Image.Image, tdset: str):
    image_name = 'sample.jpg'
    image.save(image_name)
    return_image_paths = main(tdset, image_name)
    if return_image_paths is None:
        return 'No ID object in the image.', None, None, None, None
    MSP_image = Image.open(return_image_paths['MSP'])
    Penul_SIREN_KNN_image = Image.open(return_image_paths['Penul_SIREN_KNN'])
    SAFE_SIREN_KNN_image = Image.open(return_image_paths['SAFE_SIREN_KNN'])
    SeFea_SIREN_KNN_image = Image.open(return_image_paths['SeFea_SIREN_KNN'])

    return 'Done!', MSP_image, Penul_SIREN_KNN_image, SAFE_SIREN_KNN_image, SeFea_SIREN_KNN_image

# ==== Gradio Interface ====
interface = gr.Interface(
    fn=load_and_draw_boxes,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Dropdown(
            choices=["VOC", "BDD"],
            label="ID Dataset",
            value="VOC"
        ),
    ],
    outputs=[
        gr.Text(label="Result"),
        gr.Image(type="pil", label="MSP"),
        gr.Image(type="pil", label="Penul_SIREN_KNN"),
        gr.Image(type="pil", label="SAFE_SIREN_KNN"),
        gr.Image(type="pil", label="SeFea_SIREN_KNN"),
    ],
    title="OoD Detection Demo",
    description=("Please choosing the image such that no ID object is in the image."
                "<br>ID object in VOC: " + ", ".join(voc_id_name) + "<br>ID object in BDD: " + ", ".join(bdd_id_name))
)

interface.launch()
