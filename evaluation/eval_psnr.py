# Compute PSNR between the edits on protected and unprotected images
import argparse
import pandas as pd
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchmetrics.image import PeakSignalNoiseRatio
from tqdm import trange
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import edit_prompts

psnr = PeakSignalNoiseRatio(data_range=1.0)

if __name__ == "__main__":
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_edit_dir", required=True, type=str, help="path to the directory containing edits on the unprotected images")
    parser.add_argument("--defend_edit_dirs", required=True, nargs="+", help="path to the directory containing different attack budget subdirectories")
    parser.add_argument("--seed", required=True, type=int, help="the seed to evaluate on")
    args = parser.parse_args()

    clean_edit_dir = args.clean_edit_dir
    defend_edit_dirs = args.defend_edit_dirs
    prompt_num = len(edit_prompts)
    seed = args.seed
    for x in defend_edit_dirs:
        assert os.path.exists(x)

    result = []
    
    for edit_dir in defend_edit_dirs:
        eps_dirs = sorted(os.listdir(edit_dir))
        for eps_dir in eps_dirs:
            cur_method = os.path.join(edit_dir, eps_dir)
            psnr_dict = {"method": cur_method}
            psnr_scores = []
            print(f"Processing {cur_method}")
            seed_dir = os.path.join(cur_method, f"seed{seed}")
            clean_seed_dir = os.path.join(clean_edit_dir, f"seed{seed}")
            for i in range(prompt_num):
                prompt_dir = os.path.join(seed_dir, f"prompt{i}")
                clean_prompt_dir = os.path.join(clean_seed_dir, f"prompt{i}")
                assert os.path.exists(prompt_dir) and os.path.exists(clean_prompt_dir)
                psnr_i = 0
                image_files = sorted(os.listdir(prompt_dir))
                clean_image_files = sorted(os.listdir(clean_prompt_dir))
                num = len(image_files)
                assert num == len(clean_image_files)
                for k in trange(num):
                    edit_image = Image.open(os.path.join(prompt_dir, image_files[k])).convert("RGB")
                    clean_edit_image = Image.open(os.path.join(clean_prompt_dir, clean_image_files[k])).convert("RGB")
                    edit_tensor = T.ToTensor()(edit_image)
                    clean_edit_tensor = T.ToTensor()(clean_edit_image)
                    psnr_i += psnr(edit_tensor, clean_edit_tensor).item()
                psnr_i /= num
                psnr_scores.append(psnr_i)
                psnr_dict[f"prompt{i}"] = psnr_i
            
            psnr_dict["mean"] = np.mean(psnr_scores)
            result.append(psnr_dict)

            df = pd.DataFrame(result)
            print(df)
            df.to_csv("psnr_metric.csv", index=False)
