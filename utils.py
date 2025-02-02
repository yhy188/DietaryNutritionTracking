#  等比缩放
from PIL import Image


def keep_image_size_open(path, size=(256, 256)):  # 等比例缩放图片
    img = Image.open(path)  # img读进来
    temp = max(img.size)  # 获取最长边
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))  # 根据最长边做mask掩码
    mask.paste(img, (0, 0))
    mask = mask.resize(size)  # 重构size
    return mask
