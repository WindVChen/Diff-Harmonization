import numpy as np
import torch
from PIL import Image
from typing import Optional, List
from tqdm import tqdm
from torch import optim
from attentionControl import EmptyControl, AttentionStore
from utils import view_images, aggregate_attention
import torchvision
import random


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


def diffusion_step(model, controller, latents, context, t, guidance_scale):
    latents_input = torch.cat([latents] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def preprocess(image, size):
    image = image.resize((size, size), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)[:, :3, :, :].cuda()
    return 2.0 * image - 1.0


def encoder(image, model, generator=None, size=512):
    image = preprocess(image, size)
    gpu_generator = torch.Generator(device=image.device)
    gpu_generator.manual_seed(generator.initial_seed())
    return 0.18215 * model.vae.encode(image).latent_dist.sample(generator=gpu_generator)


def text_embed_reforward(self, optim_embeddings, position):
    def forward(
            input_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        '''First step to initialize the embed, then leverage it for replacing in the following steps'''
        if optim_embeddings == [None]:
            optim_embeddings[0] = embeddings.detach()[:, position].clone()
        else:
            embeddings[:, position] = optim_embeddings

        return embeddings

    return forward


def reset_text_embed_reforward(self):
    def forward(
            input_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings

    return forward


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):

        def forward(x, context=None, mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)

            # store attention, and for following processing
            attn = controller(attn, is_cross, place_in_unet)

            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)

            # linear proj
            out = self.to_out[0](out)
            # dropout
            out = self.to_out[1](out)
            return out

        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count


def reset_attention_control(model):
    def ca_forward(self):
        def forward(x, context=None, mask=None):
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)

            # linear proj
            out = self.to_out[0](out)
            # dropout
            out = self.to_out[1](out)
            return out

        return forward

    def register_recr(net_):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_)
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                register_recr(net__)

    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            register_recr(net[1])
        elif "up" in net[0]:
            register_recr(net[1])
        elif "mid" in net[0]:
            register_recr(net[1])


@torch.no_grad()
def ddim_reverse_sample(image, prompt, model, num_inference_steps: int = 50, guidance_scale: float = 7.5,
                        generator: Optional[torch.Generator] = None, args=None):
    """
        === DDIM Inversion (See details in Section 3 `Preliminaries`) ===
    """
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    context = torch.cat(context)

    model.scheduler.set_timesteps(num_inference_steps)

    latents = encoder(image, model, generator=generator, size=args.size)
    timesteps = model.scheduler.timesteps.flip(0)

    all_latents = [latents]

    #  Not inverse the last step, as the alpha_bar_next will be set to 0 which is not aligned to its real value (~0.003)
    #  and this will lead to a bad result.
    for t in tqdm(timesteps[:-1], desc="DDIM_inverse"):
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

        next_timestep = t + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        alpha_bar_next = model.scheduler.alphas_cumprod[next_timestep] \
            if next_timestep <= model.scheduler.config.num_train_timesteps else torch.tensor(0.0)

        "leverage reversed_x0"
        reverse_x0 = (1 / torch.sqrt(model.scheduler.alphas_cumprod[t]) * (
                latents - noise_pred * torch.sqrt(1 - model.scheduler.alphas_cumprod[t])))

        latents = reverse_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * noise_pred

        all_latents.append(latents)

    #  all_latents[N] -> N: DDIM steps  (X_{T-1} ~ X_0)
    return latents, all_latents


def attention_constraint_text_optimization(prompt, model, mask, latent, inversion_latents=None,
                                           size=512, args=None):
    batch_size = len(prompt)
    mask = (torchvision.transforms.Resize([size // 32, size // 32])(mask[0, 0].unsqueeze(0)) > 100 / 255.).float()
    mask = torch.cat([mask, 1 - mask], dim=0)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0].detach()

    uncond_embeddings.requires_grad_(False)

    """Extract the text embeddings"""
    optim_embeddings = [None]

    # -2 for the added EOF token. We directly suppose that the environmental text only renders one token.
    position = len(model.tokenizer.encode(prompt[0])) - 2
    model.text_encoder.text_model.embeddings.forward = text_embed_reforward(model.text_encoder.text_model.embeddings,
                                                                            optim_embeddings, position)
    model.text_encoder(text_input.input_ids.to(model.device))[0].detach()

    basis_embeddings = optim_embeddings[0].clone()
    optim_embeddings = optim_embeddings[0].requires_grad_()

    optimizer = optim.AdamW([optim_embeddings], lr=args.op_style_lr if inversion_latents is None else args.tr_style_lr)
    loss_func = torch.nn.MSELoss()

    model.text_encoder.text_model.embeddings.forward = text_embed_reforward(model.text_encoder.text_model.embeddings,
                                                                            optim_embeddings, position)
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    context = [uncond_embeddings, text_embeddings]
    context = torch.cat(context)

    """
        For optimizing the text embeddings, we design two implementations: optimizing style and training style.
        
        === See details in Appendix B. `Implementation Details` in our paper ===
    """
    if inversion_latents is None:
        """optimizing style"""
        latent, latents = init_latent(latent, model, size, size, None, batch_size)

        # Collect all optimized text embeddings in the intermediate diffusion steps.
        intermediate_optimized_text_embed = []

        pbar = tqdm(model.scheduler.timesteps[1:], desc="Optimize_text_embed")

        #  The DDIM should begin from 1, as the inversion cannot access X_T but only X_{T-1}
        for ind, t in enumerate(pbar):
            for _ in range(args.op_style_iters):
                controller = AttentionStore()

                # Change the `forward()` in CrossAttention module of Diffusion Models.
                register_attention_control(model, controller)

                diffusion_step(model, controller, latents, context, t, 0)

                """For `loss_emb`, please refer to Eq. (3) in our paper."""
                attention_map_fg = aggregate_attention(prompt, controller, size // 32, ("up", "down"), True, 0)
                attention_map_bg = aggregate_attention(prompt, controller, size // 32, ("up", "down"), True, 1)
                attention_map = torch.stack(
                    [attention_map_fg[:, :, position] / attention_map_fg[:, :, position].max(),
                     attention_map_bg[:, :, position] / attention_map_bg[:, :, position].max()], dim=0)

                optimizer.zero_grad()
                loss_emb = loss_func(mask, attention_map.cuda())

                """For `loss_reg`, please refer to Eq. (4) in our paper."""
                loss_reg = loss_func(optim_embeddings, basis_embeddings) * args.regulation_weight
                pbar.set_postfix_str(
                    f"loss: {loss_emb.item() + loss_reg.item()}\ttext_emb_loss: {loss_emb.item()}\treg_loss: {loss_reg.item()}")
                loss = loss_emb + loss_reg

                loss.backward()
                optimizer.step()

                model.text_encoder.text_model.embeddings.forward = text_embed_reforward(
                    model.text_encoder.text_model.embeddings,
                    optim_embeddings, position)
                text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
                context = [uncond_embeddings, text_embeddings]
                context = torch.cat(context)

            intermediate_optimized_text_embed.append(text_embeddings.detach().clone())

            with torch.no_grad():
                latents = diffusion_step(model, EmptyControl(), latents, context, t, 0)

        # reset the `forward()` functions.
        reset_attention_control(model)
        model.text_encoder.text_model.embeddings.forward = reset_text_embed_reforward(
            model.text_encoder.text_model.embeddings)

        return intermediate_optimized_text_embed

    else:
        """"training style"""
        display_loss = 0
        display_reg_loss = 0
        time_list = model.scheduler.timesteps[1:]
        time_batch = args.tr_style_batch_size
        iterations = time_batch * args.tr_style_iters
        optimizer.zero_grad()
        for ind in tqdm(range(iterations)):
            # Training is similar to the default training pipeline of Diffusion Models: Random select a timestep t.
            t = random.randint(0, len(time_list) - 1)
            latent = inversion_latents[t]
            latent, latents = init_latent(latent, model, size, size, None, batch_size)

            controller = AttentionStore()

            register_attention_control(model, controller)

            diffusion_step(model, controller, latents, context, t, 0)

            """For `loss_emb`, please refer to Eq. (3) in our paper."""
            attention_map_fg = aggregate_attention(prompt, controller, size // 32, ("up", "down"), True, 0)
            attention_map_bg = aggregate_attention(prompt, controller, size // 32, ("up", "down"), True, 1)
            attention_map = torch.stack(
                [attention_map_fg[:, :, position] / attention_map_fg[:, :, position].max(),
                 attention_map_bg[:, :, position] / attention_map_bg[:, :, position].max()], dim=0)

            loss_emb = loss_func(mask, attention_map.cuda())

            """For `loss_reg`, please refer to Eq. (4) in our paper."""
            loss_reg = loss_func(optim_embeddings, basis_embeddings) * args.regulation_weight
            loss = loss_emb + loss_reg

            loss /= time_batch
            loss.backward()

            display_loss += loss.item()
            display_reg_loss += loss_reg.item()
            if (ind + 1) % time_batch == 0:
                optimizer.step()
                optimizer.zero_grad()
                print("loss: ", display_loss * time_batch / (ind + 1), "\ttext_emb_loss: ",
                      (display_loss - display_reg_loss / time_batch) * time_batch / (ind + 1), "\treg_loss: ",
                      display_reg_loss / (ind + 1))

            model.text_encoder.text_model.embeddings.forward = text_embed_reforward(
                model.text_encoder.text_model.embeddings,
                optim_embeddings, position)
            text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
            context = [uncond_embeddings, text_embeddings]
            context = torch.cat(context)

        all_text_embed = [text_embeddings.detach().clone()] * len(time_list)

        reset_attention_control(model)
        model.text_encoder.text_model.embeddings.forward = reset_text_embed_reforward(
            model.text_encoder.text_model.embeddings)
        return all_text_embed


@torch.enable_grad()
def run(
        model,
        init_prompt: List[str],
        controller,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        inversion_latents=None,
        mask=None,
        size=512,
        args=None,
):
    """
        The whole pipeline can be viewed as three steps:
        (See the overview in the beginning paragraphs of Section 4 in our paper)

        Step 1: Optimize the text conditional embeddings to ensure the text embedding can well describe the
                foreground/background environment. (See Section 4.1 in our paper)
        Step 2: Based on the optimized text embedding, optimize the unconditional embeddings to ensure the
                optimized text embeddings can reconstruct the initial image. (See Null-Text https://arxiv.org/abs/2211.09794)
                Note that Step 2 cannot be placed before Step 1, or there will be no any benefit on the content preservation.
        Step 3: Leverage the prompt-to-prompt technique (https://arxiv.org/abs/2208.01626) to harmonize the images. The
                core of this step is to fix the cross-attention maps corresponding to foreground text, and then replace
                the foreground text embeddings with the background ones. For content retention, we also fix the self-attention
                maps. (See Section 4.2 in our paper)
    """

    """Detach Diffusion Model parameters to save memory."""
    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

    """Resize the mask to the size of the latent space"""
    mask = mask.resize(latent.shape[-2:], resample=Image.LANCZOS)
    mask = np.array(mask).astype(np.float32) / 255.0
    if len(mask.shape) == 2:
        mask = mask[:, :, None]
    mask = mask[None].transpose(0, 3, 1, 2)
    mask = torch.from_numpy(mask[:, 0:1]).expand_as(latent)
    mask = (mask > 100 / 255.).float().to(latent.device)

    """
        Step 1
        
        Optimize Attention-Constraint Text Embedding to ensure the text embedding can well describe the 
        foreground/background environment.
        
        === See details in Section 4.1 `Attention-Constraint Text` in our paper. ===
    """
    constraint_text_emb = attention_constraint_text_optimization(init_prompt, model, mask, latent,
                                                                 inversion_latents=None if args.text_optimization_style == 'optimize' else inversion_latents,
                                                                 size=size, args=args)

    """
        Step 2
    
        Optimize a better unconditional embedding that can reconstruct the original image, which follows 
        `Null-text inversion for editing real images using guided diffusion models` (https://arxiv.org/abs/2211.09794).
        This is to help preserve the original image content structure.
        
        === See details in the beginning of Section 4 `Method`in our paper ===
    """
    prompt = [init_prompt[0]]  # Only use the first prompt, which describes the foreground.
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0].detach()

    uncond_embeddings.requires_grad_(True)

    optimizer = optim.AdamW([uncond_embeddings], lr=args.uncond_optimized_lr)
    loss_func = torch.nn.MSELoss()

    context = [uncond_embeddings, constraint_text_emb[0][:1]]
    context = torch.cat(context)

    latent, latents = init_latent(latent, model, size, size, generator, batch_size)

    # Collect all optimized unconditional embeddings in the intermediate diffusion steps.
    intermediate_optimized_uncond_emb = []

    #  The DDIM should begin from 1, as the inversion cannot access X_T but only X_{T-1}
    for ind, t in enumerate(tqdm(model.scheduler.timesteps[1:], desc="Optimize_uncond_embed")):
        for _ in range(ind // 10 + 1):
            out_latents = diffusion_step(model, EmptyControl(), latents, context, t, guidance_scale)

            optimizer.zero_grad()
            loss = loss_func(out_latents, inversion_latents[ind + 1])
            loss.backward()
            optimizer.step()

            context = [uncond_embeddings, constraint_text_emb[ind][:1]]
            context = torch.cat(context)

        with torch.no_grad():
            latents = diffusion_step(model, EmptyControl(), latents, context, t, guidance_scale).detach()
            intermediate_optimized_uncond_emb.append(uncond_embeddings.detach().clone())

    image = latent2image(model.vae, latents)
    view_images(image)

    """ 
        Step 3
        After getting the optimized text embeddings, we can use them to harmonize the image. The technique used here is
        prompt to prompt (https://arxiv.org/abs/2208.01626).

        === See details in the beginning of Section 4 `Method`in our paper ===
    """
    # Change the `forward()` in CrossAttention module of Diffusion Models.
    register_attention_control(model, controller)

    batch_size = len(init_prompt)

    context = [[torch.cat([intermediate_optimized_uncond_emb[i]] * batch_size), constraint_text_emb[i]] for i in
               range(len(intermediate_optimized_uncond_emb))]
    context = [torch.cat(i) for i in context]

    latent, latents = init_latent(latent, model, size, size, generator, batch_size)

    model.scheduler.set_timesteps(num_inference_steps)

    #  The DDIM should begin from 1 + start_step, as the inversion cannot access X_T but only X_{T-1}
    for ind, t in enumerate(tqdm(model.scheduler.timesteps[1:], desc="P2P_harmon_process")):
        latents = diffusion_step(model, controller, latents, context[ind], t, guidance_scale)

    image = latent2image(model.vae, latents)

    reset_attention_control(model)

    return image, latent
