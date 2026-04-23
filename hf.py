
import torch
from PIL import Image

from huggingface_hub import (
    login,
)

from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
)

def get_inputs(batch_size=1):
    generator = [torch.Generator().manual_seed(i) for i in range(batch_size)]
    prompts = batch_size * [prompt]
    num_inference_steps = 20

    return {"prompt": prompts, "generator": generator, "num_inference_steps": num_inference_steps}

def image_grid(imgs, rows=2, cols=2):
    w, h = imgs[0].size 
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

login()  # this model need auth token

model_id = "CompVis/stable-diffusion-v1-4"          # define model
pipe = StableDiffusionPipeline.from_pretrained(     # construct pipeline with SD
    model_id,
    dtype=torch.float16
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)  # sdcheduler
generator = torch.Generator().manual_seed(0)        # generator

prompt = "portait photo of a old warrior chief"
images = pipe(**get_inputs(batch_size=4)).images
grid = image_grid(images)
grid.show()