from PIL import Image, ImageChops
import os
import subprocess

def convert(directory, bmp_dir="bmp", svg_dir="svg"):
    os.makedirs(bmp_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".png"):
                src_path = os.path.join(root, f)
                bmp_path = os.path.join(bmp_dir, f[:-4] + ".bmp")
                svg_path = os.path.join(svg_dir, f[:-4] + ".svg")
                pngToBmp(src_path, bmp_path)
                # trim(bmp_path)
                bmpToSvg(bmp_path, svg_path)
                print(f"已处理: {src_path} -> {bmp_path} -> {svg_path}")

def bmpToSvg(bmp_path, svg_path):
    potrace_path = os.path.join(
        os.getcwd(), "potrace",  "potrace.exe"
    )
    subprocess.run([potrace_path, bmp_path, "-b", "svg", "-o", svg_path])

def pngToBmp(png_path, bmp_path):
    img = Image.open(png_path).convert("RGBA").resize((100, 100))
    threshold = 200
    data = []
    for pix in list(img.getdata()):
        if pix[0] >= threshold and pix[1] >= threshold and pix[3] >= threshold:
            data.append((255, 255, 255, 0))
        else:
            data.append((0, 0, 0, 1))
    img.putdata(data)
    img.save(bmp_path)

def trim(im_path):
    im = Image.open(im_path)
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    bbox = list(diff.getbbox())
    bbox[0] -= 1
    bbox[1] -= 1
    bbox[2] += 1
    bbox[3] += 1
    cropped_im = im.crop(bbox)
    cropped_im.save(im_path)

if __name__ == "__main__":
    convert("img")
