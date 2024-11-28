from PIL import Image
import torch
import argparse
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, AutoPipelineForImage2Image
import os

def get_args_parser():
    parser = argparse.ArgumentParser()

    # 1. image arguments
    parser.add_argument("--input_path", required=True, type=str, help="the path to the input image")

    # 2. edit model arguments
    parser.add_argument("--model", default="instruct-pix2pix", type=str, help="the model used to edit images [instruct-pix2pix/stable-diffusion]")
    parser.add_argument("--model_id", default="timbrooks/instruct-pix2pix", type=str, help="model id from hugging face for the model")
    parser.add_argument("--seed", default=-1, type=int, help="the seed used to control generation")
    parser.add_argument("--guidance_scale", default=7.5, type=float, help="the guidance scale for the textual prompt")
    parser.add_argument("--image_guidance_scale", default=1.5, type=float, help="the image guidance scale for the image in instruct-pix2pix")
    parser.add_argument("--strength", default=0.5, type=float, help="the strength value for the image in stable diffusion")
    parser.add_argument("--num_inference_steps", default=50, type=int, help="the number of denoising steps for image generation")
    parser.add_argument("--prompt", default="", type=str, help="the prompt used to edit the image")

    # 3. output
    parser.add_argument("--output_path", default=None, help="the path used to save the generated image")
    
    return parser

def main(args):
    # 1. prepare the image
    src_image = Image.open(args.input_path).convert("RGB")

    # 2. prepare the edit model
    model = None
    if args.model == "stable-diffusion":
        model = AutoPipelineForImage2Image.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif args.model == "instruct-pix2pix":
        model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        model.scheduler = EulerAncestralDiscreteScheduler.from_config(model.scheduler.config)
    else:
        raise ValueError(f"Invalid model '{args.model}'. Valid options are 'stable-diffusion' or 'instruct-pix2pix'.")
    
    model.to("cuda")

    # 3. edit the image
    if args.seed == -1:
        import random
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = args.seed
    torch.manual_seed(seed)
    prompt = args.prompt
    if args.model == "stable-diffusion":
        edit_image = model(
            prompt=prompt,
            image=src_image,
            num_inference_steps=args.num_inference_steps,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
        ).images[0]
    elif args.model == "instruct-pix2pix":
        edit_image = model(
            prompt=prompt,
            image=src_image,
            num_inference_steps=args.num_inference_steps,
            image_guidance_scale=args.image_guidance_scale,
            guidance_scale=args.guidance_scale,
        ).images[0]

    # 4. store the edited image
    edit_image.save(args.output_path)

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    if args.output_path is None:
        init_path = args.input_path
        directory, filename = os.path.split(init_path)
        new_path = os.path.join(directory, "edit_" + filename)
        args.output_path = new_path

    main(args)