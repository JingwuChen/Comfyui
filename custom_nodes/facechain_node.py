import sys
sys.path.append("../")
import folder_paths
import comfy
import asyncio
import os
import threading
import logging
import json
import numpy as np

from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, \
    UniPCMultistepScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, StableDiffusionXLPipeline,DiffusionPipeline
from modelscope import snapshot_download
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from torch import multiprocessing
from transformers import pipeline as tpipeline
import torch
import safetensors
from copy import deepcopy

from constants import neg_prompt, pos_prompt_with_cloth, pos_prompt_with_style
from facechain import select_high_quality_face,face_swap_fn,post_process_fn




def get_files_last_modified_time(folder_path):
    files_mtime = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and os.path.splitext(file_path)[-1] in folder_paths.supported_pt_extensions:
            files_mtime[file_path] = os.stat(file_path).st_mtime
    return files_mtime


global_model, global_clip, global_lora, global_strength_model, global_strength_clip,global_lora_path = None, None, None, None,None,None
global_model_lora, global_clip_lora = None, None
last_known_files_mtime = get_files_last_modified_time(os.path.join(folder_paths.models_dir,'loras'))
print("the file time",last_known_files_mtime)




async def hotUpdate():
    # 初始时获取文件夹内所有文件的最后修改时间
    global global_model_lora, global_clip_lora,last_known_files_mtime,global_lora_path
    global global_model, global_clip, global_lora, global_strength_model, global_strength_clip
    par_path=os.path.join(folder_paths.models_dir,'loras')
    print("the parent path is ",par_path)
    isUpdate = False
    # i=1
    while True:
       
        # 定期检查文件夹内文件是否有更新
        current_files_mtime = get_files_last_modified_time(par_path)
        # print("the current time is ",current_files_mtime,"the last",last_known_files_mtime)
        max_time=0
        for file_path, mtime in current_files_mtime.items():
            if file_path not in last_known_files_mtime or mtime > last_known_files_mtime[file_path]:
                last_known_files_mtime[file_path] = mtime
                if mtime>max_time:
                    max_time=mtime
                    global_lora_path=file_path
                isUpdate = True
        if isUpdate:
            print("the global file path is ",global_lora_path)
            global_lora = comfy.utils.load_torch_file(global_lora_path, safe_load=True) 
            model_lora, clip_lora = comfy.sd.load_lora_for_models(global_model, global_clip, global_lora, global_strength_model, global_strength_clip)
            global_model_lora, global_clip_lora = model_lora, clip_lora
            print(global_model_lora, global_clip_lora)
            isUpdate = False


class SameFileLoraLoader:
    def __init__(self):
        self.loaded_lora = None
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.start_event_loop)
        self.thread.start()
        #启动协程
        asyncio.run_coroutine_threadsafe(hotUpdate(),self.loop)


    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora_name": (folder_paths.get_filename_list("loras"), ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"


    def start_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        global global_model, global_clip, global_lora, global_strength_model, global_strength_clip,global_lora_path
        global global_model_lora, global_clip_lora
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)
        global_model, global_clip, global_lora, global_strength_model, global_strength_clip,global_lora_path = model, clip, lora, strength_model, strength_clip,lora_path
        
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        global_model_lora, global_clip_lora = model_lora, clip_lora
        
        print("the lora path is ",global_lora_path)
        return (global_model_lora, global_clip_lora)
    


class FaceChainStyleHumanSampler:
    def __init__(self):
        self.load_base_model=None
        self.loaded_style_lora = None
        self.loaded_human_lora = None
        self.init_post_model()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "base_model_path": (folder_paths.get_filename_list("facechain_base_models"),), },
                              "style_lora_name": (folder_paths.get_filename_list("facechain_style_loras"), ),
                               "human_lora_name": (folder_paths.get_filename_list("facechain_human_loras"), ),
                              "multiplier_style": ("FLOAT", {"default": 0.25, "min": -20.0, "max": 20.0, "step": 0.01}),
                              "multiplier_human": ("FLOAT", {"default": 0.85, "min": -20.0, "max": 20.0, "step": 0.01}),
                               "num_gen_images": ("INT", {"default": 6, "min": 1, "max": 4096})
                              }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"

    CATEGORY = "facechain"

    def init_post_model(self):
        post_models_dir=folder_paths.facechain_post_model_dir
        #face_quality
        if os.path.exists(os.path.join(post_models_dir,'damo/cv_manual_face-quality-assessment_fqa')):
            self.face_quality_func = pipeline(Tasks.face_quality_assessment, os.path.join(post_models_dir,
                                                                                            'damo/cv_manual_face-quality-assessment_fqa'))                  
        else:
            model_path=snapshot_download(model_id='damo/cv_manual_face-quality-assessment_fqa',revision='v2.0',cache_dir=post_models_dir)
            self.face_quality_func = pipeline(Tasks.face_quality_assessment,model=model_path)
        #face_swap
        if os.path.exists(os.path.join(post_models_dir,'damo/cv_unet_face_fusion_torch')):
            self.image_face_fusion_func = pipeline('face_fusion_torch',
                                     model=os.path.join(post_models_dir,'damo/cv_unet_face_fusion_torch'))
        else:
            model_path=snapshot_download(model_id='damo/cv_unet_face_fusion_torch',revision='v1.0.3',cache_dir=post_models_dir)
            self.image_face_fusion_func = pipeline('face_fusion_torch',
                                     model=model_path)
        if os.path.exists(os.path.join(post_models_dir,'damo/cv_ir_face-recognition-ood_rts')):
            self.face_recognition_func=pipeline(Tasks.face_recognition, os.path.join(post_models_dir,'damo/cv_ir_face-recognition-ood_rts'))
        else:
            model_path=snapshot_download(model_id='damo/cv_ir_face-recognition-ood_rts',revision='v2.5',cache_dir=post_models_dir)
            self.face_recognition_func = pipeline(Tasks.face_recognition, model=model_path)
        if os.path.exists(os.path.join(post_models_dir,'damo/cv_ddsar_face-detection_iclr23-damofd')):
            self.face_det_func=pipeline(task=Tasks.face_detection, model=os.path.join(post_models_dir,'damo/cv_ddsar_face-detection_iclr23-damofd'))
        else:
            model_path=snapshot_download(model_id='damo/cv_ddsar_face-detection_iclr23-damofd',revision='v1.1',cache_dir=post_models_dir)
            self.face_det_func = pipeline(task=Tasks.face_detection, model=model_path)
        
        

    def get_style_lora(self, style_lora_name):
       
        lora_desc_path = folder_paths.get_full_path("facechain_style_loras", style_lora_name)
        with open(lora_desc_path, "r",encoding='utf8') as f:
            lora_desc=json.load(f)
        lora_path=lora_desc['bin_file']
        if not os.path.exists(lora_path):
            raise FileExistsError('style model file do not exists ,please download first')

        add_prompt_style=lora_desc['add_prompt_style']
        style_lora=None
        if self.loaded_style_lora is not None:
            if self.loaded_style_lora[0]!=lora_path or self.loaded_style_lora[1]>os.stat(lora_path).st_mtime:
                tmp=self.loaded_style_lora
                self.loaded_style_lora=None
                del tmp
            else:
                style_lora=self.loaded_style_lora[2]
        if style_lora is None:
            if lora_path.endswith('safetensors'):
                style_lora = safetensors.load_file(lora_path)
            else:
                style_lora = torch.load(lora_path, map_location='cpu')
            self.loaded_style_lora=(lora_path,os.stat(lora_path).st_mtime,style_lora)
        return (style_lora,add_prompt_style)
    

    
    def get_generator_tag(self,json_file):
        add_prompt_style = []
        f = open(json_file, 'r',encoding='utf-8')
        tags_all = []
        cnt = 0
        cnts_trigger = np.zeros(6)
        for line in f:
            cnt += 1
            data = json.loads(line)['text'].split(', ')
            tags_all.extend(data)
            if data[1] == 'a boy':
                cnts_trigger[0] += 1
            elif data[1] == 'a girl':
                cnts_trigger[1] += 1
            elif data[1] == 'a handsome man':
                cnts_trigger[2] += 1
            elif data[1] == 'a beautiful woman':
                cnts_trigger[3] += 1
            elif data[1] == 'a mature man':
                cnts_trigger[4] += 1
            elif data[1] == 'a mature woman':
                cnts_trigger[5] += 1
            else:
                print('Error.')
        f.close()

        attr_idx = np.argmax(cnts_trigger)
        trigger_styles = ['a boy, children, ', 'a girl, children, ', 'a handsome man, ', 'a beautiful woman, ',
                        'a mature man, ', 'a mature woman, ']
        trigger_style = '(<fcsks>:10), ' + trigger_styles[attr_idx]
        if attr_idx == 2 or attr_idx == 4:
            neg_prompt += ', children'

        for tag in tags_all:
            if tags_all.count(tag) > 0.5 * cnt:
                if ('hair' in tag or 'face' in tag or 'mouth' in tag or 'skin' in tag or 'smile' in tag):
                    if not tag in add_prompt_style:
                        add_prompt_style.append(tag)


        
        if len(add_prompt_style) > 0:
            add_prompt_style = ", ".join(add_prompt_style) + ', '
        else:
            add_prompt_style = ''
        return trigger_style,add_prompt_style

    
    def get_human_lora(self, human_lora_name):
       
        lora_folder_path = folder_paths.get_full_path("human_lora_name", human_lora_name)
        lora_path=[file for file in os.listdir(lora_folder_path) if os.path.splitext(file)[-1] in folder_paths.supported_pt_extensions][0]
        if not os.path.exists(lora_path):
            raise FileNotFoundError("please train the human lora first")
        human_lora=None
        if self.loaded_human_lora is not None:
            if self.loaded_human_lora[0]!=lora_path or self.loaded_human_lora[1]>os.stat(lora_path).st_mtime \
                  or self.loaded_human_lora[2]>os.stat(lora_folder_path).st_mtime:
                tmp=self.loaded_human_lora
                self.loaded_human_lora=None
                del tmp
            else:
                human_lora=self.loaded_human_lora[3]
                trigger_style,add_prompt_style=self.loaded_human_lora[4],self.loaded_human_lora[5]

        if human_lora is None:
            if lora_path.endswith('safetensors'):
                human_lora = safetensors.load_file(lora_path)
            else:
                human_lora = torch.load(lora_path, map_location='cpu')
            
            tag_file=os.path.join(lora_folder_path, 'metadata.jsonl')
            trigger_style,add_prompt_style=self.get_generator_tag(tag_file)
            self.loaded_human_lora=(lora_path,os.stat(lora_path).st_mtime,os.stat(lora_folder_path).st_mtime,
                                    human_lora, trigger_style,add_prompt_style)

        return (human_lora,trigger_style,add_prompt_style)

    def txt2img(self,pipe, pos_prompt, neg_prompt, num_images=10, height=512, width=512, num_inference_steps=40, guidance_scale=7):
        batch_size = 5
        images_out = []
        for i in range(int(num_images / batch_size)):
            images_style = pipe(prompt=pos_prompt, height=height, width=width, guidance_scale=guidance_scale, negative_prompt=neg_prompt,
                                num_inference_steps=num_inference_steps, num_images_per_prompt=batch_size).images
            images_out.extend(images_style)
        return images_out

    def sample(self, base_model_path, style_lora_name,human_lora_name, multiplier_style, multiplier_human,
               num_gen_images,use_face_swap=True,use_post_process=True,use_stylization=False):
        pipe=None
        if self.load_base_model is not None:
            if self.load_base_model[0]!=base_model_path or self.load_base_model[1]>os.stat(base_model_path).st_mtime:
                tmp=self.load_base_model
                self.loaded_human_lora=None
                del tmp
            else:
                pipe=deepcopy(self.load_base_model[2])
        
        if pipe is None:
            pipe = DiffusionPipeline.from_pretrained(base_model_path, safety_checker=None, torch_dtype=torch.float16)
            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
            self.load_base_model=(base_model_path,os.stat(base_model_path).st_mtime,deepcopy(pipe))

        num_inference_steps = 40
        guidance_scale = 7
        #load style lora
        style_lora,pos_prompt=self.get_style_lora(style_lora_name)
        human_lora,trigger_style,add_prompt_style=self.get_human_lora(human_lora_name)
        weighted_lora_human_state_dict = {}
        for key in style_lora:
            weighted_lora_human_state_dict[key] = style_lora[key] * multiplier_human
        weighted_lora_style_state_dict = {}
        for key in human_lora:
            weighted_lora_style_state_dict[key] = human_lora[key] * multiplier_style
        
        print('start lora merging')
        pipe.load_lora_weights(weighted_lora_style_state_dict)
        print('merge style lora done')
        pipe.load_lora_weights(weighted_lora_human_state_dict)
        print('lora merging done')
        pos_prompt=trigger_style + add_prompt_style + pos_prompt
        pipe = pipe.to("cuda")
        if 'xl-base' in base_model_path:
            images_style = self.txt2img(pipe, pos_prompt, neg_prompt, num_images=10, height=768, width=768, 
                                        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
        else:
            images_style = self.txt2img(pipe, pos_prompt, neg_prompt, num_images=10, 
                                        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
        
         # select_high_quality_face PIL
        selected_face = select_high_quality_face(self.face_quality_func,human_lora_name)
        # face_swap cv2
        swap_results = face_swap_fn(use_face_swap,self.image_face_fusion_func, images_style, selected_face)
        # pose_process
        final_gen_results = post_process_fn(use_post_process, self.face_recognition_func,self.face_det_func,swap_results, selected_face,
                                       num_gen_images=num_gen_images)
        return (final_gen_results,)



        
       
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SameFileLoraLoader": SameFileLoraLoader,
    "FaceChainStyleHumanSampler":FaceChainStyleHumanSampler
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SameFileLoraLoader": "Sync SameFile Load lora",
    "FaceChainStyleHumanSampler":"Facechain inference"
}
