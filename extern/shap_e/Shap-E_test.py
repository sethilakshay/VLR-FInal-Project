import torch
from diffusers import ShapEPipeline
from diffusers.utils import export_to_gif
import pdb


ckpt_id = "openai/shap-e"
pipe = ShapEPipeline.from_pretrained(ckpt_id).to("cuda")


# prompt = "A DSLR photo of an icecream sundae inside a shopping mall."
prompt = "a cute bunny"

for step in [256, 512]:
    for guide in range(10, 110, 10):
        guidance_scale = guide
        images = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=step, frame_size = 128).images
        gif_path = export_to_gif(images[0], "./FinalRun/Bunny_Sweep/{}/Bunny_{}.gif".format(step, guide))

