import functools
import random
import os
from typing import Optional

import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont

from sample import arg_parse, load_fontdiffuser_pipeline, sampling


def load_essential_args(
    args,
    ckpt_dir: str,
    guidance_scale: float = 7.5,
):
    args.guidance_type = "classifier-free"
    args.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    args.ckpt_dir = ckpt_dir
    args.guidance_scale = guidance_scale
    return args


def run_fontdiffuser_demo_mode(
    args,
    pipe,
    ttf_path: str,
    source_image: Optional[Image.Image],
    character: str,
    reference_images,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
):

    if not character.strip():
        print("Warning: Input character string is empty.")
        return None


    output_dir = "img"
    os.makedirs(output_dir, exist_ok=True)

    if len(character) > 1:
        print(f"Processing multi-character string: '{character}'")
        generated_images = []
        current_seed = seed if isinstance(seed, int) else random.randint(0, 10000)

        base_args = args.__dict__.copy()

        for i, char in enumerate(character):
            print(f"  Generating character '{char}' ({i+1}/{len(character)})")
            char_args = type('Args', (), base_args)()

            char_args.method = "multistep"
            char_args.algorithm_type = "dpmsolver++"
            char_args.demo = True
            char_args.ttf_path = ttf_path
            char_args.character_input = False if source_image is not None else True
            char_args.content_character = char
            char_args.num_inference_steps = num_inference_steps
            char_args.guidance_scale = guidance_scale
            char_args.seed = current_seed + i

            if not isinstance(reference_images, list):
                ref_img_paths = [reference_images]
            else:
                ref_img_paths = reference_images
            style_images = [Image.open(f).convert("RGB") for f in ref_img_paths]

            try:
                char_image = sampling(
                    args=char_args,
                    pipe=pipe,
                    content_image=source_image,
                    style_images=style_images,
                )
                if char_image is not None:
                    safe_char_filename = "".join(c for c in char if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    if not safe_char_filename:
                         safe_char_filename = "unknown_char"

                    char_seed_str = str(char_args.seed)
                    import time
                    timestamp = int(time.time())
                    char_filename = f"{safe_char_filename}.png"
                    char_filepath = os.path.join(output_dir, char_filename)

                    try:
                        char_image.save(char_filepath)
                        print(f"    Saved single character image: {char_filepath}")
                    except Exception as e:
                        print(f"    Error saving single character image {char_filepath}: {e}")

                    generated_images.append(char_image)
                else:
                    print(f"    Warning: Generation failed for character '{char}', skipping.")
            except Exception as e:
                print(f"    Error generating character '{char}': {e}")
                continue

        if not generated_images:
             print("No characters were successfully generated.")
             return None

        if len(generated_images) == 1:
            final_image = generated_images[0]
        else:
            widths, heights = zip(*(img.size for img in generated_images))
            total_width = sum(widths)
            max_height = max(heights)

            final_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))
            x_offset = 0
            for img in generated_images:
                y_offset = (max_height - img.height) // 2
                final_image.paste(img, (x_offset, y_offset))
                x_offset += img.width

    else:
        print(f"Processing single character: '{character}'")
        args.method = "multistep"
        args.algorithm_type = "dpmsolver++"
        args.demo = True
        args.ttf_path = ttf_path
        args.character_input = False if source_image is not None else True
        args.content_character = character
        args.num_inference_steps = num_inference_steps
        args.guidance_scale = guidance_scale
        args.seed = seed if type(seed) is int else random.randint(0, 10000)

        if not isinstance(reference_images, list):
            reference_images = [reference_images]
        style_images = [Image.open(f).convert("RGB") for f in reference_images]

        final_image = sampling(
            args=args,
            pipe=pipe,
            content_image=source_image,
            style_images=style_images,
        )

        if final_image is not None:

            safe_char_filename = "".join(c for c in character if c.isalnum() or c in (' ', '-', '_')).rstrip()
            if not safe_char_filename:
                 safe_char_filename = "unknown_char"
            seed_str = str(args.seed)
            import time
            timestamp = int(time.time())
            char_filename = f"{safe_char_filename}_seed_{seed_str}_{timestamp}.png"
            char_filepath = os.path.join(output_dir, char_filename)

            try:
                final_image.save(char_filepath)
                print(f"Saved single character image: {char_filepath}")
            except Exception as e:
                print(f"Error saving single character image {char_filepath}: {e}")



    return final_image


def main():
    args = arg_parse()
    ckpt_dir = "ckpt"
    ttf_path = "ttf/KaiXinSongA.ttf"

    load_essential_args(
        args=args,
        ckpt_dir=ckpt_dir,
    )
    pipe = load_fontdiffuser_pipeline(args=args)

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    source_image = gr.Image(
                        width=320,
                        label="[Option 1] Source Image",
                        image_mode="RGB",
                        type="pil",
                    )
                    reference_images = gr.Files(
                        label="Reference Images",
                        type="filepath",
                        file_count="multiple"
                    )

                with gr.Row():
                    character = gr.Textbox(
                        value="何意味",
                        label="[Option 2] Source Characters (Enter multiple characters)",
                        max_lines=1
                    )
                with gr.Row():
                    fontdiffuser_output_image = gr.Image(
                        height=200,
                        label="FontDiffuser Output Image",
                        image_mode="RGB",
                        type="pil",
                    )

                num_inference_steps = gr.Slider(
                    20,
                    50,
                    value=20,
                    step=10,
                    label="Sampling Step",
                    info="The sampling step by FontDiffuser.",
                )
                guidance_scale = gr.Slider(
                    1,
                    12,
                    value=6.5,
                    step=0.5,
                    label="Scale of Classifier-free Guidance",
                    info="The scale used for classifier-free guidance sampling",
                )
                FontDiffuser = gr.Button("Run FontDiffuser")
        FontDiffuser.click(
            fn=functools.partial(run_fontdiffuser_demo_mode, args, pipe, ttf_path),
            inputs=[
                source_image,
                character,
                reference_images,
                num_inference_steps,
                guidance_scale,
            ],
            outputs=fontdiffuser_output_image,
        )
    demo.launch(debug=True)


if __name__ == "__main__":
    main()