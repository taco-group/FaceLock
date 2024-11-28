import os
from PIL import Image
import numpy as np
import torch
import torchvision
import lpips
from utils import load_model_by_repo_id, pil_to_input
from methods import cw_l2_attack, vae_attack, encoder_attack, facelock
import argparse
from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
import pdb

def get_args_parser():
    parser = argparse.ArgumentParser()

    # 1. image arguments
    parser.add_argument("--image_dir", required=True, type=str, help="the path to the images you hope to protect")

    # 2. target model arguments
    parser.add_argument("--target_model", default="instruct-pix2pix", type=str, help="the target image editing model [instruct-pix2pix/stable-diffusion]")
    parser.add_argument("--model_id", default="timbrooks/instruct-pix2pix", type=str, help="model id from hugging face for the model")
    
    # 3. attack arguments
    parser.add_argument("--defend_method", required=True, type=str, help="the chosen attack method between [encoder/vae/cw/facelock]")
    parser.add_argument("--attack_budget", default=0.03, type=float, help="the attack budget")
    parser.add_argument("--step_size", default=0.01, type=float, help="the attack step size")
    parser.add_argument("--clamp_min", default=-1, type=float, help="min value for the image pixels")
    parser.add_argument("--clamp_max", default=1, type=float, help="max value for the image pixels")
    parser.add_argument("--num_iters", default=100, type=int, help="the number of attack iterations")
    parser.add_argument("--targeted", default=True, action='store_true', help="targeted (towards 0 tensor) attack")
    parser.add_argument("--untargeted", action='store_false', dest='targeted', help="untargeted attack")

    # 3.1 cw attack other arguments
    parser.add_argument("--c", default=0.03, type=float, help="the constant ratio used in cw attack")
    parser.add_argument("--lr", default=0.03, type=float, help="the learning rate for the optimizer used in cw attack")

    # 4. output arguments
    parser.add_argument("--output_dir", required=True, type=str, help="the to the output directory")
    return parser

def main(args):
    # 1. prepare the images
    image_dir = args.image_dir
    image_files = sorted(os.listdir(image_dir))

    # 2. prepare the targeted model
    model = None
    if args.target_model == "stable-diffusion":
        model = StableDiffusionImg2ImgPipeline.from_pretrained(
            pretrained_model_name_or_path=args.model_id,
            torch_dtype=torch.float16 if args.defend_method != "cw" else torch.float32,
            safety_checker=None,
        )
    elif args.target_model == "instruct-pix2pix":
        model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            pretrained_model_name_or_path=args.model_id,
            torch_dtype=torch.float16 if args.defend_method != "cw" else torch.float32,
            safety_checker=None,
        )
        model.scheduler = EulerAncestralDiscreteScheduler.from_config(model.scheduler.config)
    else:
        raise ValueError(f"Invalid target_model '{args.target_model}'. Valid options are 'stable-diffusion' or 'instruct-pix2pix'.")
    
    model.to("cuda")

    if args.defend_method == "facelock":
        fr_id = 'minchul/cvlface_adaface_vit_base_kprpe_webface4m'
        aligner_id = 'minchul/cvlface_DFA_mobilenet'
        device = 'cuda'
        fr_model = load_model_by_repo_id(repo_id=fr_id,
                                        save_path=f'{os.environ["HF_HOME"]}/{fr_id}',
                                        HF_TOKEN=os.environ['HUGGINGFACE_HUB_TOKEN']).to(device)
        aligner = load_model_by_repo_id(repo_id=aligner_id,
                                        save_path=f'{os.environ["HF_HOME"]}/{aligner_id}',
                                        HF_TOKEN=os.environ['HUGGINGFACE_HUB_TOKEN']).to(device)
        lpips_fn = lpips.LPIPS(net="vgg").to(device)

    # 3. set up and process defend
    to_pil = torchvision.transforms.ToPILImage()
    to_tensor = torchvision.transforms.ToTensor()
    total_num = len(image_files)
    for idx, image_file in enumerate(image_files):
        if (idx + 1) % 100 == 0:
            print(f"Processing {idx + 1}/{total_num}")
        init_image = Image.open(os.path.join(image_dir, image_file)).convert("RGB")

        # process the input
        if args.defend_method == "cw":
            X = to_tensor(init_image).cuda().unsqueeze(0)
        else:
            if args.clamp_min == -1:
                X = pil_to_input(init_image).cuda().half()
            else:
                X = to_tensor(init_image).half().cuda().unsqueeze(0)

        # defend
        if args.defend_method == "encoder":
            with torch.autocast("cuda"):
                X_adv = encoder_attack(
                    X=X,
                    model=model,
                    eps=args.attack_budget,
                    step_size=args.step_size,
                    iters=args.num_iters,
                    clamp_min=-1,
                    clamp_max=1,
                    targeted=args.targeted,
                )
        elif args.defend_method == "vae":
            with torch.autocast("cuda"):
                X_adv = vae_attack(
                    X=X,
                    model=model,
                    eps=args.attack_budget,
                    step_size=args.step_size,
                    iters=args.num_iters,
                    clamp_min=-1,
                    clamp_max=1,
                )
        elif args.defend_method == "cw":
            X_adv = cw_l2_attack(
                X=X,
                model=model,
                c=args.c,
                lr=args.lr,
                iters=args.num_iters,
            )
            delta = X_adv - X
            delta_clip = delta.clip(-args.attack_budget, args.attack_budget)
            X_adv = (X + delta_clip).clip(0, 1)
        elif args.defend_method == "facelock":
            with torch.autocast("cuda"):
                X_adv = facelock(
                    X=X,
                    model=model,
                    aligner=aligner,
                    fr_model=fr_model,
                    lpips_fn=lpips_fn,
                    eps=args.attack_budget,
                    step_size=args.step_size,
                    iters=args.num_iters,
                    clamp_min=-1,
                    clamp_max=1,
                )

        if args.clamp_min == -1:
            X_adv = (X_adv / 2 + 0.5).clamp(0, 1)
        protected_image = to_pil(X_adv[0]).convert("RGB")
        protected_image.save(os.path.join(args.output_dir, image_file))

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, f"budget_{args.attack_budget}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)