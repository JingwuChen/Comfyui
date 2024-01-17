import torch
import os
import re
from collections import defaultdict

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from torch import multiprocessing
from transformers import pipeline as tpipeline
import numpy as np
from PIL import Image
import cv2
import folder_paths

def select_high_quality_face(face_quality_func,input_img_dir):
    quality_score_list = []
    abs_img_path_list = []
    ## TODO
    for img_name in os.listdir(input_img_dir):
        if img_name.endswith('jpg') or img_name.endswith('png'):
            abs_img_name = os.path.join(input_img_dir, img_name)
            face_quality_score = face_quality_func(abs_img_name)[OutputKeys.SCORES]
            if face_quality_score is None:
                quality_score_list.append(0)
            else:
                quality_score_list.append(face_quality_score[0])
            abs_img_path_list.append(abs_img_name)

    sort_idx = np.argsort(quality_score_list)[::-1]
    print('Selected face: ' + abs_img_path_list[sort_idx[0]])

    return Image.open(abs_img_path_list[sort_idx[0]])


def face_swap_fn(use_face_swap,image_face_fusion_func, gen_results, template_face):
    if use_face_swap:
        ## TODO
        out_img_list = []
        for img in gen_results:
            result = image_face_fusion_func(dict(template=img, user=template_face))[OutputKeys.OUTPUT_IMG]
            out_img_list.append(result)

        return out_img_list
    else:
        ret_results = []
        for img in gen_results:
            ret_results.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        return ret_results


def post_process_fn(use_post_process,face_recognition_func,face_det_func, swap_results_ori, selected_face, num_gen_images):
    if use_post_process:
        sim_list = []
        ## TODO
        swap_results = []
        for img in swap_results_ori:
            result_det = face_det_func(img)
            bboxes = result_det['boxes']
            if len(bboxes) == 1:
                bbox = bboxes[0]
                lenface = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                if 120 < lenface < 300:
                    swap_results.append(img)

        select_face_emb = face_recognition_func(selected_face)[OutputKeys.IMG_EMBEDDING][0]

        for img in swap_results:
            emb = face_recognition_func(img)[OutputKeys.IMG_EMBEDDING]
            if emb is None or select_face_emb is None:
                sim_list.append(0)
            else:
                sim = np.dot(emb, select_face_emb)
                sim_list.append(sim.item())
        sort_idx = np.argsort(sim_list)[::-1]

        return np.array(swap_results)[sort_idx[:min(int(num_gen_images), len(swap_results))]]
    else:
        return np.array(swap_results_ori)