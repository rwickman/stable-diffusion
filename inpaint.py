#from diffusion_img2img import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import autocast

device="cuda"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-4",
	use_auth_token=True
).to(device)

img_path = "/media/data/StochasticMachine/Videos/SD_arcane/Pics/jinx_crying_crop/frame0.png"
#mask_path = "/media/data/StochasticMachine/Videos/SD_arcane/Pics/inpaint/frame0_mask.png"
mask_path = "/media/data/StochasticMachine/Videos/SD_arcane/Pics/jinx_crop_mask.png"
prompt = "A beautiful painting of vaporwave neon light. character design by fenghua zhong, ryohei hase, ismail inceoglu and ruan jia. artstation, volumetric light, detailed, photorealistic, fantasy, rendered in octane"

def dummy(images, **kwargs):
    return images, False
pipe.safety_checker = dummy

# w, h = init_img.size

# w = w//2
# h = h//2
# print(w, h)
#w, h = map(lambda x: x - x % 64, (w, h))
init_img = Image.open(img_path).convert("RGB")
mask_img = Image.open(mask_path).convert("RGB")


#init_img = init_img.resize((768, 512))
#init_img = init_img.resize((768, 512))


with autocast("cuda"):
    generator = torch.Generator(device=device).manual_seed(0)
    out_img = pipe(
        [prompt],
        init_image=init_img,
        mask_image=mask_img,
        guidance_scale=8.5,
        num_inference_steps=30,
        strength=0.7,
        generator=generator).images[0]
out_img.save("inpaint_results_strength0_7.png")
#out_img.save(f"/media/data/StochasticMachine/Videos/SD_arcane/Pics/inpaint_results/doodle/{i}_neon_angry_strength.png")


