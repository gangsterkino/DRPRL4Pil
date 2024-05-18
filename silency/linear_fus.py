from PIL import Image

def linear_dodge(base_img_path, blend_img_path, output_img_path):
    # 打开基础图像和叠加图像
    base_image = Image.open(base_img_path).convert('RGBA')
    blend_image = Image.open(blend_img_path).convert('RGBA')
    
    if base_image.size != blend_image.size:
        blend_image = blend_image.resize(base_image.size, Image.ANTIALIAS)

    base_pixels = base_image.load()
    blend_pixels = blend_image.load()

    result_image = Image.new('RGBA', base_image.size)
    result_pixels = result_image.load()

    # Result = Base + Blend
    for y in range(base_image.size[1]):
        for x in range(base_image.size[0]):
            base_pixel = base_pixels[x, y]
            blend_pixel = blend_pixels[x, y]

            result_pixel = (
                min(base_pixel[0] + blend_pixel[0], 255),  # R
                min(base_pixel[1] + blend_pixel[1], 255),  # G
                min(base_pixel[2] + blend_pixel[2], 255),  # B
                min(base_pixel[3] + blend_pixel[3], 255)   # A
            )

            result_pixels[x, y] = result_pixel

    # 保存结果图像
    result_image.save(output_img_path)
    print(f"融合后的图像已保存到: {output_img_path}")

# 使
base_img_path = 'path/base_image.png'
blend_img_path = 'path/blend_image.png'
output_img_path = 'path/output_image.png'

linear_dodge(base_img_path, blend_img_path, output_img_path)
