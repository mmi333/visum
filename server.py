import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, make_response
from werkzeug.utils import secure_filename
from tika import parser # pip install tika
from PIL import Image

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
import os
import argparse
import torch
import random

UPLOAD_FOLDER = 'files'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda')

# Path for our main Svelte page
@app.route("/")
def base():
    return send_from_directory('client/public', 'index.html')

# Path for all the static files (compiled JS/CSS, etc.)
@app.route("/<path:path>")
def home(path):
    return send_from_directory('client/public', path)



def show_images(batch: torch.Tensor):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(torch.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    im = Image.fromarray(reshaped.numpy())
    im.save(f"client/public/images/out.png")
    return im


def text_summary(text):
    
    model_name = "snrspeaks/t5-one-line-summary"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", add_special_tokens=True)
    generated_ids = model.generate(input_ids=input_ids,num_beams=5,max_length=50,repetition_penalty=2.5,length_penalty=1,early_stopping=True,num_return_sequences=1)
    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

    one_line_sum = preds[0]
    print(one_line_sum)
    del model
    torch.cuda.empty_cache()
    imagize(one_line_sum)
    return one_line_sum


def imagize(one_line_sum):
    # Create base model.
    options = model_and_diffusion_defaults()
    options['use_fp16'] = has_cuda
    options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    if has_cuda:
        model.convert_to_fp16()
    model.to(device)
    model.load_state_dict(load_checkpoint('base', device))
    print('total base parameters', sum(x.numel() for x in model.parameters()))

    # Create upsampler model.
    options_up = model_and_diffusion_defaults_upsampler()
    options_up['use_fp16'] = has_cuda
    options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
    model_up.eval()
    if has_cuda:
        model_up.convert_to_fp16()
    model_up.to(device)
    model_up.load_state_dict(load_checkpoint('upsample', device))
    print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))
    # Sampling parameters
    prompt = one_line_sum
    batch_size = 1
    guidance_scale = 3.0
    # Tune this parameter to control the sharpness of 256x256 images.
    # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
    upsample_temp = 0.997

    ##############################
    # Sample from the base model #
    ##############################

    # Create the text tokens to feed to the model.
    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(
        tokens, options['text_ctx']
    )

    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
        [], options['text_ctx']
    )

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=torch.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        ),
        mask=torch.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=torch.bool,
            device=device,
        ),
    )

    # Create a classifier-free guidance sampling function
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    # Sample from the base model.
    model.del_cache()
    samples = diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, options["image_size"], options["image_size"]),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    model.del_cache()

    # Show the output
    show_images(samples)

    ##############################
    # Upsample the 64x64 samples #
    ##############################

    tokens = model_up.tokenizer.encode(prompt)
    tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
        tokens, options_up['text_ctx']
    )

    # Create torche model conditioning dict.
    model_kwargs = dict(
        # Low-res image to upsample.
        low_res=((samples+1)*127.5).round()/127.5 - 1,

        # Text tokens
        tokens=torch.tensor(
            [tokens] * batch_size, device=device
        ),
        mask=torch.tensor(
            [mask] * batch_size,
            dtype=torch.bool,
            device=device,
        ),
    )

    # Sample from the base model.
    model_up.del_cache()
    up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
    up_samples = diffusion_up.ddim_sample_loop(
        model_up,
        up_shape,
        noise=torch.randn(up_shape, device=device) * upsample_temp,
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    model_up.del_cache()

    # Show the output
    show_images(up_samples)



@app.route('/upload', methods=['POST'])
def upload():
    text = request.form['text']
    one_line = text_summary(text)
    return {"answer": one_line}


if __name__ == "__main__":
    app.run(debug=True)
