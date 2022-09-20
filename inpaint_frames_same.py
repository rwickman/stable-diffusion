#from diffusion_img2img import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import autocast
import os
import argparse

device="cuda"

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--load_dir", required=True, help="Directory to load frames from.")
    parser.add_argument(
        "--save_dir", required=True, help="Save directory.")
    parser.add_argument(
        "--save_frame", type=int, default=0, help="Frame to start saving.")
    parser.add_argument(
        "--start_frame", type=int, default=0, help="Frame to start load.")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed used to generate.")
    parser.add_argument(
        "--init_strength", type=float, default=0.01, help="Initial strength of inpainting.")    
    parser.add_argument(
        "--final_strength", type=float, default=1.0, help="Final strength of inpainting.")    
    parser.add_argument(
        "--mask",
        default="/media/data/StochasticMachine/Videos/SD_arcane/Pics/jinx_edits/jinx_crop_mask_full.png",
        help="Mask used for inpainting.")    
    
    
    args = parser.parse_args()

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=True
    ).to(device)


    #mask_path = "/media/data/StochasticMachine/Videos/SD_arcane/Pics/inpaint/frame0_mask.png"
    prompt = "digital drawing of glowing vaporwave neon lights swirling in space, volumetric, emissive, bloom, schizophrenic, bright"

    def dummy(images, **kwargs):
        return images, False
    pipe.safety_checker = dummy



    mask_img = Image.open(args.mask).convert("RGB")
    num_frames = len(os.listdir(args.load_dir))
    
    for i in range(num_frames):
        img_path = os.path.join(args.load_dir, f"{i + args.start_frame}_frame.png")

        init_img = Image.open(img_path).convert("RGB")
        init_img = init_img.resize((768, 512))
        cur_strength = ((num_frames - i)/num_frames) * args.init_strength + args.final_strength * (i/num_frames) 
        # print("cur_strength", cur_strength)

        with autocast("cuda"):
            generator = torch.Generator(device=device).manual_seed(args.seed)
            out_img = pipe(
                [prompt],
                init_image=init_img,
                mask_image=mask_img,
                guidance_scale=8.5,
                num_inference_steps=60,
                strength=cur_strength,
                generator=generator).images[0]

        cur_frame = os.path.join(args.save_dir, f"{i + args.save_frame}_frame.png")
        print(cur_frame)
        out_img.save(cur_frame)


#out_img.save(f"/media/data/StochasticMachine/Videos/SD_arcane/Pics/inpaint_results/doodle/{i}_neon_angry_strength.png")


