from os import makedirs
from os.path import join, exists
from typing import Final
from requests import get
from imghdr import what

from tqdm.contrib import tenumerate

name: Final = 'xyc'
root: Final = './data/Images/'
makedirs(root, exist_ok=True)

with open('./image-URLs.txt', 'r', encoding='utf-8') as f:
    image_url_set = [l.strip() for l in f.readlines()]

for i, image_url in tenumerate(image_url_set, unit='images', colour='green'):
    filepath = join(root, f'{name}{str(i).zfill(3)}.jpg')
    if exists(filepath):
        continue

    response = get(image_url)
    with open(filepath, 'wb') as image_file:
        image_file.write(response.content)

    if 'jpg' not in image_url.lower():
        real_image_type = what(filepath)
        assert real_image_type == 'jpeg', f"The image should be a JPEG, but not: {image_url} → {filepath}."
