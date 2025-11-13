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


def get_characters_from_txt(txt_file_path: str) -> str:
    """从txt文件中读取所有字符（不去重）"""
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


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

    # --- 新增：用于存储所有字符图片的列表 ---
    generated_images = []
    current_seed = seed if isinstance(seed, int) else random.randint(0, 10000)

    base_args = args.__dict__.copy()

    for i, char in enumerate(character):
        print(f"  Processing character '{char}' ({i + 1}/{len(character)})")

        # --- 新增：检查 img 文件夹下是否已有该字符的图片 ---
        safe_char_filename = "".join(c for c in char if c.isalnum() or c in (' ', '-', '_')).rstrip()
        if not safe_char_filename:
            safe_char_filename = "unknown_char"

        # 构造图片文件名，这里我们使用最简单的命名方式，不包含种子和时间戳，以便于查找
        char_filepath = os.path.join(output_dir, f"{safe_char_filename}.png")

        if os.path.exists(char_filepath):
            # 如果图片已存在，直接加载
            try:
                char_image = Image.open(char_filepath).convert("RGB")
                print(f"    Using existing image: {char_filepath}")
                generated_images.append(char_image)
                continue  # 跳过生成步骤
            except Exception as e:
                print(f"    Error loading existing image {char_filepath}: {e}")
                # 如果加载失败，继续执行生成逻辑
        # --- 新增结束 ---

        # 如果图片不存在或加载失败，则生成图片
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
        # 过滤掉 None 值和不存在的文件路径
        valid_ref_paths = [f for f in ref_img_paths if f is not None and os.path.exists(f)]
        if not valid_ref_paths:
            print("Warning: No valid reference images provided for character generation.")
            style_images = []
        else:
            style_images = [Image.open(f).convert("RGB") for f in valid_ref_paths]

        try:
            char_image = sampling(
                args=char_args,
                pipe=pipe,
                content_image=source_image,
                style_images=style_images,
            )
            if char_image is not None:
                # 保存图片，使用简单文件名
                try:
                    char_image.save(char_filepath)
                    print(f"    Saved new image: {char_filepath}")
                    generated_images.append(char_image)
                except Exception as e:
                    print(f"    Error saving new image {char_filepath}: {e}")
            else:
                print(f"    Warning: Generation failed for character '{char}', skipping.")
        except Exception as e:
            print(f"    Error generating character '{char}': {e}")
            continue

    if not generated_images:
        print("No characters were successfully processed (generated or loaded).")
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

    return final_image


def process_with_mode(
        args,
        pipe,
        ttf_path: str,
        source_image: Optional[Image.Image],
        input_mode: str,  # 'Manual Input' or 'Upload TXT File'
        manual_character: str,
        txt_file: str,
        reference_images,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
):
    """根据选择的模式处理请求"""
    print(f"DEBUG: input_mode = {input_mode}, manual_character = '{manual_character}', txt_file = {txt_file}")
    if input_mode == 'Upload TXT File' and txt_file is not None:
        # --- 关键修改：只在 'Upload TXT File' 模式且文件存在时，才读取txt文件 ---
        try:
            # 从txt文件中读取字符
            all_characters = get_characters_from_txt(txt_file.name)
            # 只取前10个字符用于生成和显示
            characters_to_generate = all_characters[:10]
            print(f"Processing characters from TXT file (first 10 shown): '{characters_to_generate}'")
        except Exception as e:
            print(f"Error reading TXT file: {e}")
            print("Failed to read TXT file, aborting generation.")
            return None
    elif input_mode == 'Manual Input':
        # 使用手动输入的字符
        characters_to_generate = manual_character
        print(f"Processing manually entered characters: '{characters_to_generate}'")
    else:
        # 如果是 'Upload TXT File' 模式但没有上传文件，或者模式无效
        print(f"No valid input provided or file not uploaded in 'Upload TXT File' mode. input_mode: {input_mode}, txt_file: {txt_file}")
        return None

    # 调用原来的生成函数
    result = run_fontdiffuser_demo_mode(
        args, pipe, ttf_path, source_image, characters_to_generate,
        reference_images, num_inference_steps, guidance_scale, seed
    )

    return result


def process_style_transfer(
        args,
        pipe,
        ttf_path: str,
        source_image: Image.Image,  # 必须是单字图像
        reference_images,           # 风格参考图像
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
):
    """处理风格迁移模式：源图像是单字图像，参考图像是风格参考"""
    if source_image is None:
        print("Error: Source image is required for style transfer.")
        return None

    # 尝试从源图像中推断字符（如果可能），或者让用户指定字符（这里简化，假设字符是 'X' 或者从其他地方获取）
    # 为了兼容 sampling 函数，我们需要一个 content_character
    # 一个简单的办法是让用户输入一个字符，或者从源图像中推断（这很复杂）
    # 为了演示，我们假设字符是 'X'，但实际上 sampling 应该能处理 content_image 为非 None 的情况
    # 关键在于 sampling 函数如何使用 content_image 和 style_images
    # 我们传递一个空字符串或单个占位符字符，并设置 character_input=False
    # 但这可能需要 sampling 函数内部的逻辑支持
    # 如果 sampling 函数设计为：有 content_image 时，忽略 content_character，那么我们可以传入一个占位符
    # 但更可能的是，我们需要确保 sampling 函数接收正确的参数
    # 在 run_fontdiffuser_demo_mode 中，我们已经设置了 args.character_input = False if source_image is not None else True
    # 这意味着如果 source_image 存在，sampling 应该使用 content_image 而不是 content_character

    # 为了风格迁移，我们只需要一个字符，可以是任意字符或从源图像推断
    # 这里我们假设一个占位符字符，因为 content_image 会覆盖 content_character
    dummy_character = "X" # 占位符，sampling 函数应优先使用 source_image

    # 重用 run_fontdiffuser_demo_mode，但只处理一个字符
    # 设置 character_input 为 False，表示使用 content_image
    args.character_input = False
    args.content_character = dummy_character # 占位符
    args.method = "multistep"
    args.algorithm_type = "dpmsolver++"
    args.demo = True
    args.ttf_path = ttf_path
    args.num_inference_steps = num_inference_steps
    args.guidance_scale = guidance_scale
    args.seed = seed if type(seed) is int else random.randint(0, 10000)

    # 处理参考图像
    if not isinstance(reference_images, list):
        reference_images = [reference_images]
    valid_ref_paths = [f for f in reference_images if f is not None and os.path.exists(f)]
    if not valid_ref_paths:
        print("Warning: No valid reference images provided for style transfer.")
        style_images = []
    else:
        style_images = [Image.open(f).convert("RGB") for f in valid_ref_paths]

    try:
        result_image = sampling(
            args=args,
            pipe=pipe,
            content_image=source_image, # 关键：传入源图像
            style_images=style_images,  # 关键：传入风格图像
        )
        if result_image is not None:
            # 保存结果
            output_dir = "img"
            os.makedirs(output_dir, exist_ok=True)
            safe_char_filename = "style_transfer_result" # 或者基于源图像文件名生成
            import time
            timestamp = int(time.time())
            result_filename = f"{safe_char_filename}_{timestamp}.png"
            result_filepath = os.path.join(output_dir, result_filename)
            try:
                result_image.save(result_filepath)
                print(f"Saved style transfer result: {result_filepath}")
            except Exception as e:
                print(f"Error saving style transfer result {result_filepath}: {e}")
        return result_image
    except Exception as e:
        print(f"Error during style transfer: {e}")
        return None


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
        # 添加顶层选项卡
        with gr.Tabs():
            with gr.TabItem("Font Image to Font Image (Style Transfer)"):
                # --- Option 1: Style Transfer ---
                with gr.Row():
                    with gr.Column(scale=1):
                        source_image_opt1 = gr.Image(
                            width=320,
                            label="Source Image (Single Character)",
                            image_mode="RGB",
                            type="pil",
                        )
                        # 可以添加一个输入框让用户指定源图像中的字符（可选）
                        # source_char_opt1 = gr.Textbox(
                        #     label="Character in Source Image (Optional)",
                        #     max_lines=1
                        # )

                        with gr.Row():
                            fontdiffuser_output_image_opt1 = gr.Image(
                                height=200,
                                label="Style Transfer Output Image",
                                image_mode="RGB",
                                type="pil",
                            )

                        num_inference_steps_opt1 = gr.Slider(
                            20,
                            50,
                            value=20,
                            step=10,
                            label="Sampling Step",
                            info="The sampling step by FontDiffuser.",
                        )
                        guidance_scale_opt1 = gr.Slider(
                            1,
                            12,
                            value=6.5,
                            step=0.5,
                            label="Scale of Classifier-free Guidance",
                            info="The scale used for classifier-free guidance sampling",
                        )

                        FontDiffuser_opt1 = gr.Button("Run Style Transfer")

                    # 右侧列：参考图像
                    with gr.Column(scale=1):
                        reference_images_opt1 = gr.Files(
                            label="Reference Images (Style)",
                            type="filepath",
                            file_count="multiple"
                        )

                FontDiffuser_opt1.click(
                    fn=functools.partial(process_style_transfer, args, pipe, ttf_path),
                    inputs=[
                        source_image_opt1,
                        reference_images_opt1,
                        num_inference_steps_opt1,
                        guidance_scale_opt1,
                    ],
                    outputs=fontdiffuser_output_image_opt1,
                )

            with gr.TabItem("Direct TTF as Source"):
                # --- Option 2: TTF Source (Original functionality) ---
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            source_image_opt2 = gr.Image(
                                width=320,
                                label="[Option 1] Source Image (Optional)",
                                image_mode="RGB",
                                type="pil",
                            )
                            reference_images_opt2 = gr.Files(
                                label="Reference Images",
                                type="filepath",
                                file_count="multiple"
                            )

                        # 添加模式切换组件
                        input_mode_opt2 = gr.Radio(
                            choices=["Manual Input", "Upload TXT File"],
                            value="Manual Input",
                            label="Input Mode",
                            interactive=True
                        )

                        # 创建两个输入框，根据模式显示其中一个
                        with gr.Group(visible=True) as manual_input_group_opt2:
                            manual_character_opt2 = gr.Textbox(
                                value="何意味", # 这是方式1的默认值
                                label="[Option 2] Source Characters (Enter multiple characters)",
                                max_lines=1
                            )

                        with gr.Group(visible=False) as txt_input_group_opt2:
                            txt_file_opt2 = gr.File(
                                label="Upload TXT File (Will generate first 10 unique characters)",
                                file_types=[".txt"],
                                type="filepath"
                            )

                        # 定义切换函数
                        def switch_input_mode_opt2(mode):
                            if mode == "Manual Input":
                                return gr.update(visible=True), gr.update(visible=False)
                            else:  # Upload TXT File
                                return gr.update(visible=False), gr.update(visible=True)

                        # 绑定切换事件
                        input_mode_opt2.change(
                            fn=switch_input_mode_opt2,
                            inputs=input_mode_opt2,
                            outputs=[manual_input_group_opt2, txt_input_group_opt2]
                        )

                        with gr.Row():
                            fontdiffuser_output_image_opt2 = gr.Image(
                                height=200,
                                label="FontDiffuser Output Image",
                                image_mode="RGB",
                                type="pil",
                            )

                        num_inference_steps_opt2 = gr.Slider(
                            20,
                            50,
                            value=20,
                            step=10,
                            label="Sampling Step",
                            info="The sampling step by FontDiffuser.",
                        )
                        guidance_scale_opt2 = gr.Slider(
                            1,
                            12,
                            value=6.5,
                            step=0.5,
                            label="Scale of Classifier-free Guidance",
                            info="The scale used for classifier-free guidance sampling",
                        )

                        FontDiffuser_opt2 = gr.Button("Run FontDiffuser")

                        # 修改按钮点击事件，传入模式选择
                        FontDiffuser_opt2.click(
                            fn=functools.partial(process_with_mode, args, pipe, ttf_path),
                            inputs=[
                                source_image_opt2, # 这个在 TTF 模式下通常为 None
                                input_mode_opt2,  # 新增：输入模式
                                manual_character_opt2,  # 手动输入字符
                                txt_file_opt2,  # txt文件
                                reference_images_opt2,
                                num_inference_steps_opt2,
                                guidance_scale_opt2,
                            ],
                            outputs=fontdiffuser_output_image_opt2,
                        )

    demo.launch(debug=True)


if __name__ == "__main__":
    main()