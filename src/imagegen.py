import config
import cv2
from torch import autocast
import numpy as np
import torch
import gc

class ImageGen():

    def __init__(self, model):
        self.model = model.to("cuda")

    def prompt2image(self, text):
        prompt = (text)
        with autocast("cuda"):
            image = self.model(prompt, guidance_scale=7.5)["sample"][0]
        imagearr = np.array(image)
        cv2.imwrite(config.OUT_PATH+f"{prompt}.jpg", imagearr)
        return image
    
    def img2image(self, image, text):
        init_img = image
        prompt = (text)
        with autocast("cuda"):
            torch.cuda.empty_cache()    
            gc.collect()
            image = self.model(prompt=prompt, init_image=init_img, strength=0.5, guidance_scale=7.5)["sample"][0]
            imagearr = np.array(image)
            cv2.imwrite(config.OUT_PATH+f"img2img{prompt}.jpg", imagearr)
            return image
