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
    parser.add_argument("--input_path", required=True, type=str, help="the path to the image you hope to protect")

    # 2. target model arguments
    parser.add_argument("--target_model", default="instruct-pix2pix", type=str, help="the target image editing model [instruct-pix2pix/stable-diffusion]")
    parser.add_argument("--model_id", default="timbrooks/instruct-pix2pix", type=str, help="model id from hugging face for the model")
    
    # 3. attack arguments
    parser.add_argument("--defend_method", required=True, type=str, help="the chosen attack method between [encoder/vae/cw/facelock]")
    parser.add_argument("--attack_budget", default=0.03, type=float, help="the attack budget")
    parser.add_argument("--step_size", default=0.01, type=float, help="the attack step size")
    parser.add_argument("--num_iters", default=100, type=int, help="the number of attack iterations")
    parser.add_argument("--targeted", default=True, action='store_true', help="targeted (towards 0 tensor) attack")
    parser.add_argument("--untargeted", action='store_false', dest='targeted', help="untargeted attack")

    # 3.1 cw attack other arguments
    parser.add_argument("--c", default=0.03, type=float, help="the constant ratio used in cw attack")
    parser.add_argument("--lr", default=0.03, type=float, help="the learning rate for the optimizer used in cw attack")

    # 4. output arguments
    parser.add_argument("--output_path", default=None, type=str, help="the output path the protected images")
    return parser

def process_encoder_attack(X, model, args):
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
    return X_adv

def process_vae_attack(X, model, args):
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
    return X_adv

def process_cw_attack(X, model, args):
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
    return X_adv

def process_facelock(X, model, args):
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
    return X_adv

def main(args):
    # 1. prepare the image
    init_image = Image.open(args.input_path).convert("RGB")
    to_tensor = torchvision.transforms.ToTensor()
    if args.defend_method != "cw":
        X = pil_to_input(init_image).cuda().half()
    else:
        X = to_tensor(init_image).cuda().unsqueeze(0)   # perform cw attack using torch.float32

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

    # 3. set up defend
    defend_fn = None
    if args.defend_method == "encoder":
        defend_fn = process_encoder_attack
    elif args.defend_method == "vae":
        defend_fn = process_vae_attack
    elif args.defend_method == "cw":
        defend_fn = process_cw_attack
    elif args.defend_method == "facelock":
        defend_fn = process_facelock
    else:
        raise ValueError(f"Invalid defend_method '{args.defend_method}'. Valid options are 'encoder', 'vae', 'cw', or 'facelock'.")
    
    # 4. process defend
    X_adv = defend_fn(X, model, args)

    # 5. convert back to image and store it
    to_pil = torchvision.transforms.ToPILImage()
    if args.defend_method != "cw":
        X_adv = (X_adv / 2 + 0.5).clamp(0, 1)
    protected_image = to_pil(X_adv[0]).convert("RGB")
    protected_image.save(args.output_path)

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)