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
        source_image: Optional[Image.Image], # 保留，用于 Option 1
        character_for_display: str, # 用于显示的字符（前10个）
        all_characters_to_cache: str, # 用于缓存的所有字符
        reference_images,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
):
    if not character_for_display.strip():
        print("Warning: Character string for display is empty.")
        return None

    output_dir = "img"
    os.makedirs(output_dir, exist_ok=True)

    # --- 新增：首先处理所有需要缓存的字符 ---
    current_seed = seed if isinstance(seed, int) else random.randint(0, 10000)
    base_args = args.__dict__.copy()

    # 确保 reference_images 被处理
    if not isinstance(reference_images, list):
        reference_images = [reference_images]
    valid_ref_paths = [f for f in reference_images if f is not None and os.path.exists(f)]
    if not valid_ref_paths:
        print("Warning: No valid reference images provided for character generation.")
        style_images = []
    else:
        style_images = [Image.open(f).convert("RGB") for f in valid_ref_paths]

    for i, char in enumerate(all_characters_to_cache):
        print(f"  Caching character: '{char}' ({i + 1}/{len(all_characters_to_cache)})")

        # --- 检查 img 文件夹下是否已有该字符的图片 ---
        safe_char_filename = "".join(c for c in char if c.isalnum() or c in (' ', '-', '_')).rstrip()
        if not safe_char_filename:
            safe_char_filename = "unknown_char"

        # 构造图片文件名
        char_filepath = os.path.join(output_dir, f"{safe_char_filename}.png")

        if os.path.exists(char_filepath):
            # 如果图片已存在，跳过生成步骤
            print(f"    Using existing image for cache: {char_filepath}")
            continue  # 跳过生成步骤
        else:
            # 如果图片不存在，则生成图片并保存
            char_args = type('Args', (), base_args)()

            char_args.method = "multistep"
            char_args.algorithm_type = "dpmsolver++"
            char_args.demo = True
            char_args.ttf_path = ttf_path
            # Option 2 时 source_image 为 None，所以 character_input 会是 True
            char_args.character_input = False if source_image is not None else True
            char_args.content_character = char
            char_args.num_inference_steps = num_inference_steps
            char_args.guidance_scale = guidance_scale
            char_args.seed = current_seed + i # 使用当前字符的索引作为种子偏移

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
                        print(f"    Saved new image for cache: {char_filepath}")
                    except Exception as e:
                        print(f"    Error saving new image for cache {char_filepath}: {e}")
                else:
                    print(f"    Warning: Generation failed for character '{char}' (for cache), skipping.")
            except Exception as e:
                print(f"    Error generating character '{char}' (for cache): {e}")
                continue
    # --- 缓存处理结束 ---

    # --- 现在处理用于显示的字符（character_for_display） ---
    generated_images = []
    for i, char in enumerate(character_for_display):
        print(f"  Processing character for display: '{char}' ({i + 1}/{len(character_for_display)})")

        # --- 检查 img 文件夹下是否已有该字符的图片 ---
        safe_char_filename = "".join(c for c in char if c.isalnum() or c in (' ', '-', '_')).rstrip()
        if not safe_char_filename:
            safe_char_filename = "unknown_char"

        # 构造图片文件名
        char_filepath = os.path.join(output_dir, f"{safe_char_filename}.png")

        if os.path.exists(char_filepath):
            # 如果图片已存在，直接加载
            try:
                char_image = Image.open(char_filepath).convert("RGB")
                print(f"    Using existing image for display: {char_filepath}")
                generated_images.append(char_image)
                continue  # 跳过生成步骤
            except Exception as e:
                print(f"    Error loading existing image for display {char_filepath}: {e}")
                # 如果加载失败，继续执行生成逻辑
        else:
            # 理论上不应该走到这里，因为上面已经缓存了所有需要的字符
            # 但如果因为某些原因文件不存在，我们再生成一次
            print(f"    Image not found in cache for display, attempting to generate: {char_filepath}")
            char_args = type('Args', (), base_args)()

            char_args.method = "multistep"
            char_args.algorithm_type = "dpmsolver++"
            char_args.demo = True
            char_args.ttf_path = ttf_path
            # Option 2 时 source_image 为 None，所以 character_input 会是 True
            char_args.character_input = False if source_image is not None else True
            char_args.content_character = char
            char_args.num_inference_steps = num_inference_steps
            char_args.guidance_scale = guidance_scale
            char_args.seed = current_seed + i # 使用当前字符的索引作为种子偏移

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
                        print(f"    Saved new image for display: {char_filepath}")
                        generated_images.append(char_image)
                    except Exception as e:
                        print(f"    Error saving new image for display {char_filepath}: {e}")
                        # 如果保存失败，可以选择不添加到列表，或者添加生成的图像
                        generated_images.append(char_image) # 仍然添加到列表用于显示
                else:
                    print(f"    Warning: Generation failed for character '{char}' (for display), skipping.")
            except Exception as e:
                print(f"    Error generating character '{char}' (for display): {e}")
                continue

    if not generated_images:
        print("No characters were successfully processed (generated or loaded) for display.")
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
        input_mode: str,  # 'Manual Input' or 'Upload TXT File'
        manual_character: str,
        txt_file: str,
        reference_images,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
):
    """根据选择的模式处理请求 (Option 2 专用)"""
    print(f"DEBUG: input_mode = {input_mode}, manual_character = '{manual_character}', txt_file = {txt_file}")
    # 在 Option 2 中，source_image 始终为 None
    source_image = None

    if input_mode == 'Upload TXT File' and txt_file is not None:
        # --- 关键修改：只在 'Upload TXT File' 模式且文件存在时，才读取txt文件 ---
        try:
            # 从txt文件中读取**所有**字符
            all_characters = get_characters_from_txt(txt_file.name)
            # 只取前10个字符用于**显示**
            characters_to_display = all_characters[:10]
            print(f"Processing characters from TXT file (first 10 shown): '{characters_to_display}'")
            print(f"Will cache images for ALL characters in TXT file: '{all_characters}'")
        except Exception as e:
            print(f"Error reading TXT file: {e}")
            print("Failed to read TXT file, aborting generation.")
            return None
    elif input_mode == 'Manual Input':
        # 使用手动输入的字符
        characters_to_display = manual_character
        all_characters = manual_character # 对于手动输入，所有字符就是显示的字符
        print(f"Processing manually entered characters: '{characters_to_display}'")
    else:
        # 如果是 'Upload TXT File' 模式但没有上传文件，或者模式无效
        print(f"No valid input provided or file not uploaded in 'Upload TXT File' mode. input_mode: {input_mode}, txt_file: {txt_file}")
        return None

    # 调用原来的生成函数，传入所有字符用于缓存，前10个字符用于显示
    result = run_fontdiffuser_demo_mode(
        args, pipe, ttf_path, source_image, characters_to_display, all_characters, # source_image 为 None, characters_to_display, all_characters
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

    # 为了兼容 sampling 函数，我们需要一个 content_character
    # 我们传递一个占位符字符，并设置 character_input=False
    dummy_character = "X" # 占位符，sampling 函数应优先使用 source_image

    # 设置参数
    args.character_input = False
    args.content_character = dummy_character
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
            safe_char_filename = "style_transfer_result"
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
                        with gr.Row():
                            source_image_opt1 = gr.Image(
                                width=320,
                                label="Source Image (Single Character)",
                                image_mode="RGB",
                                type="pil",
                            )
                            # 将参考图像上传和预览放在同一行
                            with gr.Column(scale=1): # 这个 Column 用于包裹 Row
                                with gr.Row():
                                    # 显示第一个参考图像的预览
                                    reference_image_preview_opt1 = gr.Image(
                                        width=320,
                                        label="Reference Image Preview (First)",
                                        image_mode="RGB",
                                        type="pil",
                                        interactive=False # 设置为不可交互，仅用于预览
                                    )
                                    # 上传多个参考图像
                                    reference_files_opt1 = gr.Files(
                                        label="Upload Reference Images (Style)",
                                        type="filepath",
                                        file_count="multiple"
                                    )

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

                        # 定义函数：当上传文件改变时，更新预览图像
                        def update_reference_preview_opt1(files):
                            if files and len(files) > 0:
                                # 加载第一个文件
                                try:
                                    img = Image.open(files[0]).convert("RGB")
                                    return img
                                except Exception as e:
                                    print(f"Error loading preview image: {e}")
                                    return None
                            else:
                                return None

                        # 绑定 change 事件
                        reference_files_opt1.change(
                            fn=update_reference_preview_opt1,
                            inputs=reference_files_opt1,
                            outputs=reference_image_preview_opt1
                        )

                # 绑定点击事件，将 reference_files_opt1 作为参考图像输入
                FontDiffuser_opt1.click(
                    fn=functools.partial(process_style_transfer, args, pipe, ttf_path),
                    inputs=[
                        source_image_opt1,
                        reference_files_opt1, # 传入整个文件列表
                        num_inference_steps_opt1,
                        guidance_scale_opt1,
                    ],
                    outputs=fontdiffuser_output_image_opt1,
                )

            with gr.TabItem("Direct TTF as Source"):
                # --- Option 2: TTF Source (Original functionality, without source image) ---
                with gr.Row():
                    with gr.Column(scale=1):
                        # 移除了 source_image_opt2
                        # 将参考图像上传和预览放在同一行
                        with gr.Column(scale=1): # 这个 Column 用于包裹 Row
                            with gr.Row():
                                # 显示第一个参考图像的预览
                                reference_image_preview_opt2 = gr.Image(
                                    width=320,
                                    height=400,
                                    label="Reference Image Preview (First)",
                                    image_mode="RGB",
                                    type="pil",
                                    interactive=False  # 设置为不可交互，仅用于预览
                                )
                                # 上传多个参考图像
                                reference_images_opt2 = gr.Files(
                                    label="Upload Reference Images",
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

                        # 定义函数：当上传文件改变时，更新预览图像
                        def update_reference_preview_opt2(files):
                            if files and len(files) > 0:
                                # 加载第一个文件
                                try:
                                    img = Image.open(files[0]).convert("RGB")
                                    return img
                                except Exception as e:
                                    print(f"Error loading preview image: {e}")
                                    return None
                            else:
                                return None

                        # 绑定 change 事件
                        reference_images_opt2.change(
                            fn=update_reference_preview_opt2,
                            inputs=reference_images_opt2,
                            outputs=reference_image_preview_opt2
                        )

                        # 修改按钮点击事件，移除 source_image_opt2 输入
                        FontDiffuser_opt2.click(
                            fn=functools.partial(process_with_mode, args, pipe, ttf_path),
                            inputs=[
                                # source_image_opt2, # 移除
                                input_mode_opt2,  # 新增：输入模式
                                manual_character_opt2,  # 手动输入字符
                                txt_file_opt2,  # txt文件
                                reference_images_opt2, # 传入整个文件列表
                                num_inference_steps_opt2,
                                guidance_scale_opt2,
                            ],
                            outputs=fontdiffuser_output_image_opt2,
                        )

    demo.launch(debug=True)


if __name__ == "__main__":
    main()