import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import utils
from attentionControl import AttentionReplace
import diff_harmon
from PIL import Image
import numpy as np
import os
import argparse
import glob
from natsort import ns, natsorted


def run_harmonization(image, prompts, diffusion_model, diffusion_steps, guidance=7.5, generator=None, device='cpu',
                      cross_replace_steps=1., self_replace_steps=1., init_guidance=0, mask=None, size=512,
                      save_dir="./output", args=None):
    os.makedirs(save_dir, exist_ok=True)
    bg = np.array(image.resize((size, size), resample=Image.LANCZOS))[:, :, :3]
    m = (np.array(mask.resize((size, size), resample=Image.LANCZOS)) > 100).astype(np.float32)
    if len(m.shape) == 2:
        m = m[:, :, None]
    elif m.shape[2] != 1:
        m = m[:, :, 0:1]

    for ind in range(10):
        print(f"\n======================================================\n"
              f"=================      Iteration:{ind}      ==================\n"
              f"======================================================")

        """Do DDIM inversion. Collect all the intermediate latents in the inverse steps."""
        init_image = image
        init_prompt = [prompts[0]]
        x_t, inversion_latents = diff_harmon.ddim_reverse_sample(init_image, init_prompt, diffusion_model,
                                                                 diffusion_steps,
                                                                 init_guidance, generator, args=args)

        """Do the Diffusion Harmonization."""
        controller = AttentionReplace(prompts, diffusion_model.tokenizer, diffusion_steps,
                                      cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps,
                                      device=device)
        out_img, _ = diff_harmon.run(diffusion_model, prompts, controller, latent=x_t,
                                     num_inference_steps=diffusion_steps,
                                     guidance_scale=guidance, generator=generator,
                                     inversion_latents=inversion_latents[::-1], mask=mask, size=size, args=args)

        """Visualize the attention maps and the final results."""
        utils.show_cross_attention(prompts, diffusion_model.tokenizer, controller, res=size // 32,
                                   from_where=("up", "down"),
                                   save_path="{}/{}_repeat_attentionFG.jpg".format(save_dir, str(ind).rjust(2, '0')))
        utils.show_self_attention_comp(prompts, controller, res=size // 32, from_where=("up", "down"),
                                       save_path="{}/{}_repeat_selfAttention.jpg".format(save_dir,
                                                                                         str(ind).rjust(2, '0')))

        # Here, we don't visualize the attention map corresponding to background text, as its attention map has been
        # replaced by foreground's in `diff_harmon.py`.
        # utils.show_cross_attention(prompts, diffusion_model.tokenizer, controller, res=size // 32, from_where=("up", "down"),
        #                            save_path="{}/{}_repeat_attentionBG.jpg".format(save_dir,
        #                                                                             str(ind).rjust(2, '0')), select=1)

        ori = Image.fromarray(out_img[-1].astype(np.uint8))
        ori.save("{}/{}_repeat_ori.jpg".format(save_dir, str(ind).rjust(2, '0')))

        image = m * out_img[-1] + (1 - m) * bg

        image = Image.fromarray(image.astype(np.uint8))
        image.save("{}/{}_repeat_blend.jpg".format(save_dir, str(ind).rjust(2, '0')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', default="./output", type=str,
                        help='Where to save the results')
    parser.add_argument('--pretrained_diffusion_path',
                        default="stabilityai/stable-diffusion-2-base",
                        type=str,
                        help='Set the path to `stabilityai/stable-diffusion-2-base`.')
    parser.add_argument('--harmonize_iterations', default=10, type=int, help='How many times to harmonize the images')
    parser.add_argument('--is_single_image', action='store_true', help='Whether to test on a single image or images')

    # For single image
    parser.add_argument('--image_path', default="./demo/girl_comp.jpg", type=str)
    parser.add_argument('--mask_path', default="./demo/girl_mask.jpg", type=str)
    parser.add_argument('--foreground_prompt', default="girl autumn", type=str,
                        help='Text describes the environment of foreground.')
    parser.add_argument('--background_prompt', default="girl winter", type=str,
                        help='Text describes the environment of background.')

    # For multiple images
    parser.add_argument('--images_root', default="./demo/composite", type=str,
                        help='The composite images root directory')
    parser.add_argument('--masks_root', default="./demo/mask", type=str, )
    parser.add_argument('--caption_txt', default="./demo/caption.txt", type=str, help='The caption txt file')

    # Hyperparameters
    parser.add_argument('--seed', default=8888, type=int, help='Random seed')
    parser.add_argument('--diffusion_steps', default=50, type=int, help='Total DDIM sampling steps')
    parser.add_argument('--guidance', default=2.5, type=float, help='guidance scale of diffusion models')
    parser.add_argument('--size', default=512, type=int, help='The input image resized size')

    parser.add_argument('--uncond_optimized_lr', default=1e-1, type=float,
                        help='Learning rate for optimizing unconditional embeddings.')

    parser.add_argument('--text_optimization_style', default='optimize', type=str, choices=['optimize', 'train'],
                        help='Whether using optimizing style or training style for text embedding optimization.')
    parser.add_argument('--regulation_weight', default=1000, type=int, help='Regulation loss weight.')

    # Hyperparameters for text embedding optimizing style
    parser.add_argument('--op_style_lr', default=1e-3, type=float,
                        help='Learning rate for optimization style of text embedding.')
    parser.add_argument('--op_style_iters', default=2, type=int, help='Optimization iterations.')

    # Hyperparameters for text embedding training style
    parser.add_argument('--tr_style_lr', default=1e-2, type=float,
                        help='Learning rate for training style of text embedding.')
    parser.add_argument('--tr_style_iters', default=50, type=int, help='Optimization iterations.')
    parser.add_argument('--tr_style_batch_size', default=4, type=int, help='Batch size.')

    args = parser.parse_args()

    generator = torch.Generator().manual_seed(args.seed)
    diffusion_steps = args.diffusion_steps
    guidance = args.guidance
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    ldm_stable = StableDiffusionPipeline.from_pretrained(args.pretrained_diffusion_path).to(device)
    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)

    if args.is_single_image:
        "Test on a single image"
        composite_image = Image.open(args.image_path)
        mask = Image.open(args.mask_path)

        prompts = [args.foreground_prompt, args.background_prompt]
        run_harmonization(composite_image, prompts, ldm_stable, diffusion_steps, guidance=guidance, generator=generator,
                          device=device, mask=mask, size=args.size, save_dir=args.save_dir, args=args)
    else:
        "Test on multiple images"
        composite_images = []
        mask_images = []
        for i in glob.glob(os.path.join(args.images_root, "*")):
            composite_images.append(i)
        for i in glob.glob(os.path.join(args.masks_root, "*")):
            mask_images.append(i)
        composite_images = natsorted(composite_images, alg=ns.PATH)
        mask_images = natsorted(mask_images, alg=ns.PATH)

        with open(args.caption_txt, "r") as f:
            data = f.readlines()
            captions = []
            for i in data:
                cap = i.rstrip().split(",")
                captions.append(cap)

        for ind, img in enumerate(composite_images):
            prefix = img.split("\\")[-1][:-4]
            composite_image = Image.open(img)
            mask = Image.open(mask_images[ind])
            prompts = captions[ind]
            run_harmonization(composite_image, prompts, ldm_stable, diffusion_steps, guidance=guidance,
                              generator=generator, device=device, mask=mask,
                              save_dir=os.path.join(args.save_dir, prefix), size=args.size, args=args)
