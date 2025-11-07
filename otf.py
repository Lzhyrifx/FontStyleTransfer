import fontforge
import os

# 配置
IMG_DIR = "img/"              # 图片目录
FONT_NAME = "MyBitmapFont"
FONT_FAMILY = "My Bitmap Font"
FONT_FULLNAME = "My Bitmap Font"

# 创建新字体
font = fontforge.font()
font.fontname = FONT_NAME
font.familyname = FONT_FAMILY
font.fullname = FONT_FULLNAME
font.copyright = "Created from images by user"

# 设置字体度量（可选）
font.ascent = 800
font.descent = 200
font.em = 1000

# 遍历图片
for filename in os.listdir(IMG_DIR):
    if not filename.endswith(".png"):
        continue
    try:
        # 解析文件名获取 Unicode 码点
        codepoint_str = filename.split(".")[0]
        codepoint = int(codepoint_str, 16)
    except ValueError:
        print(f"跳过无效文件: {filename}")
        continue

    # 创建字形并导入位图
    glyph = font.createChar(codepoint)
    img_path = os.path.join(IMG_DIR, filename)

    # 导入位图（自动缩放至 em 单位）
    glyph.importBitmaps(img_path)

    # 设置字形宽度（默认为 em 单位）
    glyph.width = 1000

    print(f"已导入: {chr(codepoint)} (U+{codepoint:04X})")

# 保存为 .sfd（FontForge 项目文件，便于后续编辑）
font.save("myfont.sfd")

# 生成 TTF（嵌入位图）
font.generate("myfont.ttf", flags=("bitmap",))

print("\n✅ TTF 字体生成完成: myfont.ttf")