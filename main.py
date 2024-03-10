import os
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import utils
from attentionControl import AttentionReplace
import diff_harmon
from PIL import Image
import numpy as np
import argparse
import glob
from natsort import ns, natsorted
from PIL import ImageFile
from HarmonizationDetect.inference import harmon_detect
import random
import shutil
ImageFile.LOAD_TRUNCATED_IMAGES = True


def run_harmonization_no_evaluator(image, prompts, diffusion_model, diffusion_steps, guidance=7.5, generator=None, device='cpu',
                      cross_replace_steps=1., self_replace_steps=1., init_guidance=0, mask=None, size=512,
                      save_dir="./output", args=None):
    os.makedirs(save_dir, exist_ok=True)
    bg = np.array(image.resize((size, size), resample=Image.LANCZOS))[:, :, :3]
    m = (np.array(mask.resize((size, size), resample=Image.LANCZOS)) > 100).astype(np.float32)
    if len(m.shape) == 2:
        m = m[:, :, None]
    elif m.shape[2] != 1:
        m = m[:, :, 0:1]

    for ind in range(args.harmonize_iterations):
        print(f"\n======================================================\n"
              f"===============      Iteration:{ind}      ================\n"
              f"======================================================")

        """Do DDIM inversion. Collect all the intermediate latents in the inverse steps."""
        init_image = image
        init_prompt = [prompts[0][0]]
        x_t, inversion_latents = diff_harmon.ddim_reverse_sample(init_image, init_prompt, diffusion_model,
                                                                 diffusion_steps,
                                                                 init_guidance, generator, args=args)

        """Do the Diffusion Harmonization."""
        controller = AttentionReplace(prompts[0], diffusion_model.tokenizer, diffusion_steps,
                                      cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps,
                                      device=device)
        out_img, _ = diff_harmon.run(diffusion_model, prompts[0], controller, latent=x_t,
                                     num_inference_steps=diffusion_steps,
                                     guidance_scale=guidance, generator=generator,
                                     inversion_latents=inversion_latents[::-1], mask=mask, size=size, args=args,
                                     original_image=image)

        """Visualize the attention maps and the final results."""
        # utils.show_cross_attention(prompts, diffusion_model.tokenizer, controller, res=size // 32,
        #                            from_where=("up", "down"),
        #                            save_path="{}/{}_repeat_attentionFG.jpg".format(save_dir, str(ind).rjust(2, '0')))
        # utils.show_self_attention_comp(prompts, controller, res=size // 32, from_where=("up", "down"),
        #                                save_path="{}/{}_repeat_selfAttention.jpg".format(save_dir,
        #                                                                                  str(ind).rjust(2, '0')))
        #
        # ori = Image.fromarray(out_img[-1].astype(np.uint8))
        # ori.save("{}/{}_repeat_ori.jpg".format(save_dir, str(ind).rjust(2, '0')))

        image = m * out_img[-1] + (1 - m) * bg

        image = Image.fromarray(image.astype(np.uint8))
        image.save("{}/{}_repeat_blend.jpg".format(save_dir, str(ind).rjust(2, '0')))

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

    """
        Below are the processes of the Performance Evaluation.
        (Details can be found in Section 3.3 in our paper-v2)
    """
    args.prompt_num = len(prompts)
    args.prompt_change_flag = None
    harmonization_scores = dict((i, []) for i in range(args.prompt_num))
    for ind in range(args.harmonize_iterations):
        print(f"\n======================================================\n"
              f"===============      Iteration:{ind}      ================\n"
              f"======================================================")

        init_image = image
        if ind == 0:
            # At first, calculate harmonization score for each prompt
            for idx in range(args.prompt_num):
                init_prompt = [prompts[idx][0]]
                """Do DDIM inversion. Collect all the intermediate latents in the inverse steps."""
                x_t, inversion_latents = diff_harmon.ddim_reverse_sample(init_image, init_prompt, diffusion_model,
                                                                         diffusion_steps,
                                                                         init_guidance, generator, args=args)

                """Do the Diffusion Harmonization."""
                controller = AttentionReplace(prompts[idx], diffusion_model.tokenizer, diffusion_steps,
                                              cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps,
                                              device=device)
                out_img, _ = diff_harmon.run(diffusion_model, prompts[idx], controller, latent=x_t,
                                             num_inference_steps=diffusion_steps,
                                             guidance_scale=guidance, generator=generator,
                                             inversion_latents=inversion_latents[::-1], mask=mask, size=size, args=args,
                                             original_image=image)

                """Visualize the attention maps and the final results."""
                # utils.show_cross_attention(prompts[idx], diffusion_model.tokenizer, controller, res=size // 32,
                #                            from_where=("up", "down"),
                #                            save_path="{}/{}_repeat_attentionFG_prompt{}.jpg".format(save_dir,
                #                                                             str(ind).rjust(2, '0'),str(idx).rjust(2, '0')))
                # utils.show_self_attention_comp(prompts[idx], controller, res=size // 32, from_where=("up", "down"),
                #                                save_path="{}/{}_repeat_selfAttention_prompt{}.jpg".format(save_dir,
                #                                                             str(ind).rjust(2, '0'),str(idx).rjust(2, '0')))
                #
                # ori = Image.fromarray(out_img[-1].astype(np.uint8))
                # ori.save("{}/{}_repeat_ori_prompt{}.jpg".format(save_dir, str(ind).rjust(2, '0'),str(idx).rjust(2, '0')))

                image = m * out_img[-1] + (1 - m) * bg

                image = Image.fromarray(image.astype(np.uint8))
                image.save("{}/{}_repeat_blend_prompt{}.jpg".format(save_dir, str(ind).rjust(2, '0'),str(idx).rjust(2, '0')))

            for idx in range(args.prompt_num):
                score = harmon_detect("{}/{}_repeat_blend_prompt{}.jpg".format(save_dir, str(ind).rjust(2, '0'),
                                                                         str(idx).rjust(2, '0')),args.mask_path)
                harmonization_scores[idx].append(score)
            print("The harmonization scores of each prompt in the first iteration are: {}".format(harmonization_scores))
            # Among the initially generated several prompts, choose the prompt with the highest score.
            args.max_score = max(harmonization_scores.values())
            print("The prompt with the highest score is: {}".format(args.max_score))
            args.max_score_prompt = max(harmonization_scores, key=harmonization_scores.get)
            print("The prompt with the highest score is: {}".format(prompts[args.max_score_prompt]))
            image = Image.open("{}/{}_repeat_blend_prompt{}.jpg".format(save_dir, str(ind).rjust(2, '0'),str(args.max_score_prompt).rjust(2, '0')))

        else:
            if args.prompt_change_flag is not None:
                prompt_used_idx = args.prompt_change_flag
            else:
                prompt_used_idx = args.max_score_prompt
            init_prompt = [prompts[prompt_used_idx][0]]
            """Do DDIM inversion. Collect all the intermediate latents in the inverse steps."""
            x_t, inversion_latents = diff_harmon.ddim_reverse_sample(init_image, init_prompt, diffusion_model,
                                                                     diffusion_steps,
                                                                     init_guidance, generator, args=args)

            """Do the Diffusion Harmonization."""
            controller = AttentionReplace(prompts[prompt_used_idx], diffusion_model.tokenizer, diffusion_steps,
                                          cross_replace_steps=cross_replace_steps,
                                          self_replace_steps=self_replace_steps,
                                          device=device)
            out_img, _ = diff_harmon.run(diffusion_model, prompts[prompt_used_idx], controller, latent=x_t,
                                         num_inference_steps=diffusion_steps,
                                         guidance_scale=guidance, generator=generator,
                                         inversion_latents=inversion_latents[::-1], mask=mask, size=size, args=args,
                                         original_image=image)

            """Visualize the attention maps and the final results."""
            # utils.show_cross_attention(prompts[prompt_used_idx], diffusion_model.tokenizer, controller, res=size // 32,
            #                            from_where=("up", "down"),
            #                            save_path="{}/{}_repeat_attentionFG_prompt{}.jpg".format(save_dir,
            #                                                                         str(ind).rjust(2,'0'),
            #                                                                        str(prompt_used_idx).rjust(2,'0')))
            # utils.show_self_attention_comp(prompts[prompt_used_idx], controller, res=size // 32, from_where=("up", "down"),
            #                                save_path="{}/{}_repeat_selfAttention_prompt{}.jpg".format(save_dir,str(ind).rjust(2, '0'),
            #                                                             str(prompt_used_idx).rjust(2, '0')))

            # ori = Image.fromarray(out_img[-1].astype(np.uint8))
            # ori.save("{}/{}_repeat_ori_prompt{}.jpg".format(save_dir, str(ind).rjust(2, '0'), str(prompt_used_idx).rjust(2, '0')))

            image = m * out_img[-1] + (1 - m) * bg

            image = Image.fromarray(image.astype(np.uint8))
            image.save(
                "{}/{}_repeat_blend_prompt{}.jpg".format(save_dir, str(ind).rjust(2, '0'), str(prompt_used_idx).rjust(2, '0')))

            # Leverage the evaluator (a lightweight classifier) to calculate the harmonization score.
            score = harmon_detect("{}/{}_repeat_blend_prompt{}.jpg".format(save_dir, str(ind).rjust(2, '0'),
                                                                             str(prompt_used_idx).rjust(2, '0')), args.mask_path)
            harmonization_scores[prompt_used_idx].append(score)

            for i in range(args.prompt_num):
                if i != prompt_used_idx:
                    harmonization_scores[i].append(0)

            # If decrease three times, then regenerate (change) the prompt, or stop.
            if len(harmonization_scores[prompt_used_idx]) > 2:
                if harmonization_scores[prompt_used_idx][-1] < harmonization_scores[prompt_used_idx][-2] < \
                        harmonization_scores[prompt_used_idx][-3]:
                    if args.prompt_change_flag is not None:
                        scores = [item for sublist in list(harmonization_scores.values()) for item in sublist]
                        final_max_score = max(scores)
                        print("The prompt with the highest score is: {}".format(final_max_score))
                        for key, value in harmonization_scores.items():
                            if final_max_score in harmonization_scores[key]:
                                final_max_score_prompt = key
                                final_ind = harmonization_scores[key].index(final_max_score)
                                break
                        print("The prompt is: {}".format(prompts[final_max_score_prompt]))
                        # copy the image and rename it
                        shutil.copy("{}/{}_repeat_blend_prompt{}.jpg".format(save_dir, str(final_ind).rjust(2, '0'),
                                                                                  str(final_max_score_prompt).rjust(2, '0')),
                                     "{}/final_output.jpg".format(save_dir))
                        break
                    else:
                        """
                        Regenerate (change) prompt idx. In case potential network connection error, 
                        here we provide an offline way to directly use the pre-generated mutliple prompts. 
                        You can easily adapt it back to the online way (as dicted in our paper-v2) by referring 
                        to our `gemini_mini_vision.py`.
                        """
                        prompts_id_list = list(harmonization_scores.keys())
                        prompts_id_list.remove(args.max_score_prompt)
                        if len(prompts_id_list) == 0:
                            break
                        args.prompt_change_flag = random.choice(prompts_id_list)
                        print("\n ==== Change prompt to: ", args.prompt_change_flag, prompts[args.prompt_change_flag])

                        "Back to the prev-best status."
                        image = Image.open(
                            "{}/{}_repeat_blend_prompt{}.jpg".format(save_dir, str(ind - 2).rjust(2, '0'),
                                                                     str(prompt_used_idx).rjust(2, '0')))

    # get the max score prompt
    scores = [item for sublist in list(harmonization_scores.values()) for item in sublist]
    final_max_score = max(scores)
    print("The prompt with the highest score is: {}".format(final_max_score))
    for key,value in harmonization_scores.items():
        if final_max_score in harmonization_scores[key]:
            final_max_score_prompt = key
            final_ind = harmonization_scores[key].index(final_max_score)
            break
    print("The prompt is: {}".format(prompts[final_max_score_prompt]))
    shutil.copy(
        "{}/{}_repeat_blend_prompt{}.jpg".format(save_dir, str(final_ind).rjust(2, '0'),
                                                   str(final_max_score_prompt).rjust(2, '0')),
        "{}/final_output.jpg".format(save_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', default="./output", type=str,
                        help='Where to save the results')
    parser.add_argument('--pretrained_diffusion_path',
                        default="stabilityai/stable-diffusion-2-base",
                        type=str,
                        help='Set the path to `stabilityai/stable-diffusion-2-base`.')
    parser.add_argument('--harmonize_iterations', default=10, type=int, help='How many times to harmonize the images')
    parser.add_argument('--is_single_image', action='store_false', help='Whether to test on a single image or images')
    parser.add_argument('--use_edge_map', action='store_true', help='Whether to use edge maps')
    parser.add_argument('--use_evaluator', action='store_true', help='Whether to automatically pick results')

    # For single image
    parser.add_argument('--image_path', default="./demo/girl_comp.jpg", type=str)
    parser.add_argument('--mask_path', default="./demo/girl_mask.jpg", type=str)
    parser.add_argument('--foreground_prompt', default="girl golden autumn", type=str,
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

    parser.add_argument('--regulation_weight', default=1000, type=int, help='Regulation loss weight.')

    # Hyperparameters for text embedding optimizing style
    parser.add_argument('--op_style_lr', default=5e-4, type=float,
                        help='Learning rate for optimization style of text embedding.')
    parser.add_argument('--op_style_iters', default=2, type=int, help='Optimization iterations.')

    args = parser.parse_args()

    generator = torch.Generator().manual_seed(args.seed)
    diffusion_steps = args.diffusion_steps
    guidance = args.guidance
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    ldm_stable = StableDiffusionPipeline.from_pretrained(args.pretrained_diffusion_path).to(device)
    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)

    harmon_fun = run_harmonization if args.use_evaluator else run_harmonization_no_evaluator

    if args.is_single_image:
        "Test on a single image"
        composite_image = Image.open(args.image_path)
        mask = Image.open(args.mask_path)

        prompts = [[args.foreground_prompt, args.background_prompt]]
        harmon_fun(composite_image, prompts, ldm_stable, diffusion_steps, guidance=guidance, generator=generator,
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
                cap = i.rstrip().split(";")
                c_list = []
                for i in range(len(cap)):
                    c = cap[i].split(",")
                    c_list.append(c)
                captions.append(c_list)

        for ind, img in enumerate(composite_images):
            prefix = img.split("/")[-1][:-4]
            composite_image = Image.open(img)
            mask = Image.open(mask_images[ind])
            args.mask_path = mask_images[ind]
            prompts = captions[ind]
            harmon_fun(composite_image, prompts, ldm_stable, diffusion_steps, guidance=guidance,
                              generator=generator, device=device, mask=mask,
                              save_dir=os.path.join(args.save_dir, prefix), size=args.size, args=args)
