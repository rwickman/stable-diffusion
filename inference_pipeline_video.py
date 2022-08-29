from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
import torch
from PIL import Image
import numpy as np
from scipy.spatial import geometric_slerp
import os
# 1. Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token="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")

# 2. Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_auth_token="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")



torch_device = "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device) 


#prompt = ["a detailed photo of a vaporwave, colorful, trippy, multidimensional being that is a ghost high-quality, ArtStation"]
#prompt = ["family guy peter griffin high-quality, ArtStation meme", "family guy peter griffin meme simpsons"]
#prompt = ["a detailed photo of a vaporwave, colorful, trippy, multidimensional being that is a ghost high-quality, ArtStation", "neon multidimensional trippy ghost"]

# prompt_1 = ["cosmic horror, abstract, ghostly, arcade, duotone, poltergeist, elegant, highly detailed, artstation, smooth, sharp focus, unreal engine 5, raytracing, art by beeple and mike winkelmann, ultraviolet colors"]
prompt = ["a detailed photo of the end of the universe, artstation, vaporwave, neon", "a detailed photo of the end of the universe with a blackhole, artstation, vaporwave, neon"]

#prompt_2 = ["galaxy, multidimensional, abstract, ghostly, poltergeist, elegant, highly detailed, artstation, smooth, sharp focus, unreal engine 5, raytracing, art by beeple and mike winkelmann, ultraviolet colors"]
# prompt_3 = ["a highly detailed black hole with starts swirling poltergeist artstation unreal engine 5 raytracing art by beeple and mike winkelmann, ultraviolet colors"]
# prompt_4 = ["a highly detailed black hole with starts swirling poltergeist artstation raytracing ultraviolet colors"]
save_dir = "results_3/16_ghost_lerp_longer"
name_prefix = "ghost_slerp"

# Make save directory
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


height = 512                        # default height of Stable Diffusion
width = 768                         # default width of Stable Diffusion
num_interp_steps = 101
num_inference_steps = 60           # Number of denoising steps

#guidance_scale = 7.5               # Scale for classifier-free guidance
guidance_scale = 8.0
slerp = False
#generator = None
batch_size = 1#len(prompt)

if slerp:
    start = np.array([1, 0])
    end = np.array([0, 1])
    t_vals = np.linspace(0, 1, num_interp_steps)
    interp_factors_arr = geometric_slerp(start, end, t_vals)
    interp_factors = [i[0] for i in interp_factors_arr]
else:
    interp_factors = np.linspace(0, 1, num_interp_steps)[::-1]
    print(interp_factors)


for img_idx, interp_factor in enumerate(interp_factors):
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    generator = torch.manual_seed(0)    # Seed generator to create the inital latent noise
    print(f"interp_factor {interp_factor}")
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    max_length = text_input.input_ids.shape[-1]
    # uncond_input = tokenizer(
    #     [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    # )
    uncond_input = tokenizer(
        [""] * 1, padding="max_length", max_length=max_length, return_tensors="pt"
    )

    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]   
    

    text_embeddings = interp_factor * text_embeddings[0] + (1 - interp_factor) * text_embeddings[1]
    
    text_embeddings = text_embeddings.unsqueeze(0)
    
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(torch_device)

    scheduler.set_timesteps(num_inference_steps)

    latents = latents * scheduler.sigmas[0]

    from tqdm.auto import tqdm


    # with autocast("cuda"):
    for i, t in tqdm(enumerate(scheduler.timesteps)):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, i, latents)["prev_sample"]


    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)

    # convert the image to PIL so we can display or save it.
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save(f"{save_dir}/{img_idx}_{name_prefix}_{round(interp_factor, 3)}.png")