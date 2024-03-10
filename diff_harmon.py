import numpy as np
import torch
from PIL import Image
from typing import Optional, List
from tqdm import tqdm
from torch import optim
from attentionControl import EmptyControl, AttentionStore
from utils import view_images, aggregate_attention
import torchvision
from thirdparty.edge_detector import get_edge, Initialize_PidNet, pidnet_args, PidNet
from torchvision import transforms


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


def latent2image_edge_fusion(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    # normalize the image: mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]
    image_edge = image.squeeze(dim=0)
    image_edge = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_edge)
    image_edge = image_edge.unsqueeze(dim=0)
    return image, image_edge


def preprocess(image, size):
    image = image.resize((size, size), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)[:, :3, :, :].cuda()
    return 2.0 * image - 1.0


def preprocess_pidnet(image, size):
    image = image.convert('RGB')
    transform = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    image = transform(image)
    image = image.unsqueeze(dim=0).cuda()
    return image


def encoder(image, model, generator=None, size=512):
    image = preprocess(image, size)
    gpu_generator = torch.Generator(device=image.device)
    gpu_generator.manual_seed(generator.initial_seed())
    return 0.18215 * model.vae.encode(image).latent_dist.sample(generator=gpu_generator)


def text_embed_reforward(self, optim_embeddings, position_fg, position_bg):
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
        fg_emb, bg_emb = torch.chunk(embeddings, 2, dim=0)
        if optim_embeddings == [None, None]:
            optim_embeddings[0] = [fg_emb.detach()[:, x].clone() for x in position_fg]
            optim_embeddings[1] = [bg_emb.detach()[:, x].clone() for x in position_bg]
        else:
            for ind, x in enumerate(position_fg):
                embeddings[:1, x] = optim_embeddings[0][ind]
            for ind, x in enumerate(position_bg):
                embeddings[1:, x] = optim_embeddings[1][ind]

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
        === DDIM Inversion ===
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


def attention_constraint_text_optimization(prompt, model, mask, latent, size=512, args=None):
    """
    Text Embedding Refinement design. Please refer to Section 3.2 in our paper-v2 for more information.
    """
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
    optim_embeddings = [None, None]

    position_start = len(model.tokenizer.encode(prompt[0].split()[0])) - 1
    position_end_fg = len(model.tokenizer.encode(prompt[0])) - 2
    position_end_bg = len(model.tokenizer.encode(prompt[1])) - 2
    position_fg = [x for x in range(position_start, position_end_fg + 1)]
    position_bg = [x for x in range(position_start, position_end_bg + 1)]
    model.text_encoder.text_model.embeddings.forward = text_embed_reforward(model.text_encoder.text_model.embeddings,
                                                                            optim_embeddings, position_fg, position_bg)
    model.text_encoder(text_input.input_ids.to(model.device))[0].detach()

    basis_embeddings_fg = torch.cat(optim_embeddings[0]).clone()
    basis_embeddings_bg = torch.cat(optim_embeddings[1]).clone()
    optim_embeddings_fg = [x.requires_grad_() for x in optim_embeddings[0]]
    optim_embeddings_bg = [x.requires_grad_() for x in optim_embeddings[1]]

    weight_emb_bg = torch.nn.Parameter(torch.ones(len(position_bg), device=model.device) / len(position_bg))

    optimizer = optim.AdamW(optim_embeddings_fg + optim_embeddings_bg + [weight_emb_bg],
                            lr=args.op_style_lr)
    loss_func = torch.nn.MSELoss()

    model.text_encoder.text_model.embeddings.forward = text_embed_reforward(model.text_encoder.text_model.embeddings,
                                                                            [optim_embeddings_fg, optim_embeddings_bg],
                                                                            position_fg, position_bg)
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    context = [uncond_embeddings, text_embeddings]
    context = torch.cat(context)

    """
        For optimizing the text embeddings, we design two implementations: optimizing style and training style. We here
        only present optimizing style.
    """

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

            """For loss function calculation, please refer to Eq. 2~4 in our paper."""
            attention_map_fg = aggregate_attention(prompt, controller, size // 32, ("up", "down"), True, 0).cuda()
            attention_map_bg = aggregate_attention(prompt, controller, size // 32, ("up", "down"), True, 1).cuda()
            optimizer.zero_grad()
            loss_emb = 0
            attention_map = 0
            for ind, idd in enumerate(position_fg):
                attention_map += attention_map_fg[:, :, idd]

            loss_emb += loss_func(mask[0], (attention_map / attention_map.max()))

            attention_map = 0
            for ind, idd in enumerate(position_bg):
                attention_map += attention_map_bg[:, :, idd] * weight_emb_bg[ind]

            loss_emb += loss_func(mask[1], (attention_map / attention_map.max()))

            """For `loss_reg`, please refer to Eq. (3) in our paper."""
            loss_reg = (loss_func(torch.cat(optim_embeddings_fg), basis_embeddings_fg) + loss_func(
                torch.cat(optim_embeddings_bg), basis_embeddings_bg)) * args.regulation_weight
            pbar.set_postfix_str(
                f"loss: {loss_emb.item() + loss_reg.item()}\ttext_emb_loss: {loss_emb.item()}\treg_loss: {loss_reg.item()}")
            loss = loss_emb + loss_reg

            loss.backward()
            optimizer.step()

            model.text_encoder.text_model.embeddings.forward = text_embed_reforward(
                model.text_encoder.text_model.embeddings,
                optim_embeddings, position_fg, position_bg)
            text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
            context = [uncond_embeddings, text_embeddings]
            context = torch.cat(context)

            with torch.no_grad():
                weight_emb_bg.data = torch.nn.functional.softmax(weight_emb_bg, dim=0).data

        fuse_emb = (text_embeddings[1, position_bg, :].detach().clone()) * weight_emb_bg.unsqueeze(1)
        fuse_emb = fuse_emb.sum(0).unsqueeze(0)
        text_embeddings[1, position_bg, :] = fuse_emb

        intermediate_optimized_text_embed.append(text_embeddings.detach().clone())

        with torch.no_grad():
            latents = diffusion_step(model, EmptyControl(), latents, context, t, 0)

    # reset the `forward()` functions.
    reset_attention_control(model)
    model.text_encoder.text_model.embeddings.forward = reset_text_embed_reforward(
        model.text_encoder.text_model.embeddings)

    return intermediate_optimized_text_embed


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
        original_image=None,
):
    """
        Below are the scripts related to Section 3.2 in our paper-v2

        Step 1: Optimize the text conditional embeddings to ensure the text embedding can well describe the
                foreground/background environment. (See `Text Embedding Refinement` in our Sec. 3.2)
        Step 2: Based on the optimized text embedding, optimize the unconditional embeddings to align the edge maps from
                sobel and PidNet and also ensure the optimized text embeddings can reconstruct the initial image. (See
                Null-Text https://arxiv.org/abs/2211.09794) Note that Step 2 cannot be placed before Step 1, or there
                will be no any benefit on the content preservation. (See `Content Structure Preservation` in our Sec. 3.2)
        Step 3: Leverage the prompt-to-prompt technique (https://arxiv.org/abs/2208.01626) to harmonize the images. The
                core of this step is to fix the cross-attention maps corresponding to foreground text, and then replace
                the foreground text embeddings with the background ones. For content retention, we also fix the self-attention
                maps. (See `Foreground Editing` and `Content Structure Preservation` in our Sec. 3.2)
    """

    """Detach Diffusion Model parameters to save memory."""
    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

    """Original mask"""
    original_mask = mask.resize((size, size), resample=Image.LANCZOS)
    original_mask = np.array(original_mask).astype(np.float32) / 255.0
    if len(original_mask.shape) == 2:
        original_mask = original_mask[:, :, None]
    original_mask = original_mask[None].transpose(0, 3, 1, 2)
    original_mask = torch.from_numpy(original_mask[:, 0:1])
    original_mask = (original_mask > 100 / 255.).float().to(latent.device)

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
        
        Optimize Text Embedding to ensure the text embedding can well describe the foreground/background environment.
        
        === See details in Section 3.2 `Text Embedding Refinement` in our paper-v2. ===
    """
    constraint_text_emb = attention_constraint_text_optimization(init_prompt, model, mask, latent,
                                                                 size=size, args=args)

    """
        Step 2
    
        Optimize a better unconditional embedding that can align the edge maps from sobel and PidNet and also reconstruct 
        the original image. The optimization follows `Null-text inversion for editing real images using guided diffusion models` 
        (https://arxiv.org/abs/2211.09794).
        
        === See details in the beginning of Section 3.2 `Content Structure Preservation`in our paper-v2 ===
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

    # Calculate the original's edge map.
    pidnet = Initialize_PidNet(pidnet_args)
    original_edge = PidNet(pidnet, preprocess_pidnet(original_image, 512))
    original_sobel = get_edge(preprocess(original_image, 512))

    #  The DDIM should begin from 1, as the inversion cannot access X_T but only X_{T-1}
    for ind, t in enumerate(tqdm(model.scheduler.timesteps[1:], desc="Optimize_uncond_embed")):
        for _ in range(ind // 10 + 1):
            out_latents = diffusion_step(model, EmptyControl(), latents, context, t, guidance_scale)

            image_sobel, image_edge = latent2image_edge_fusion(model.vae, out_latents)

            # Calculate the harmonized image's edge maps.
            out_edge = PidNet(pidnet, image_edge)
            out_sobel = get_edge(image_sobel)

            optimizer.zero_grad()
            loss_null_text = loss_func(out_latents, inversion_latents[ind + 1])
            loss_edge_all = 0.1 * loss_func(out_edge * original_mask, original_edge * original_mask) + loss_func(
                out_sobel * original_mask, original_sobel * original_mask)
            loss = 10 * loss_null_text + (loss_edge_all if args.use_edge_map else 0)

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

        === See details in `Foreground Editing` of Section 3.2 in our paper-v2 ===
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
