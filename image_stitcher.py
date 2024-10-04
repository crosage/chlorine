import os
import re
from PIL import Image

input_folder_path = r'D:\yrcc2\img_dir\train_output'
output_folder_path = r'D:\yrcc2\img_dir\output'

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

pattern = re.compile(r'([\d_]+)_(LTR|RTL)_h(\d)_v(\d)\.(jpg|tif)')

ltr_images = {}
rtl_images = {}
image_extensions = {}

for file_name in os.listdir(input_folder_path):
    if (file_name.endswith('.jpg') or file_name.endswith('.tif')) and not file_name.endswith('_stitched.jpg'):
        match = pattern.match(file_name)
        if match:
            image_id = match.group(1)
            direction = match.group(2)
            h_value = int(match.group(3))
            v_value = int(match.group(4))
            extension = match.group(5)

            img_path = os.path.join(input_folder_path, file_name)

            image_extensions[image_id] = extension

            if direction == 'LTR':
                if image_id not in ltr_images:
                    ltr_images[image_id] = []
                ltr_images[image_id].append((h_value, v_value, img_path))
            elif direction == 'RTL':
                if image_id not in rtl_images:
                    rtl_images[image_id] = []
                rtl_images[image_id].append((h_value, v_value, img_path))

def stitch_images(image_list, reverse_horizontal=False, reverse_vertical=False):
    image_list.sort(key=lambda x: (x[0], x[1]))

    if reverse_vertical:
        image_list.sort(key=lambda x: (x[0], -x[1]))

    unique_h_values = sorted(set([x[0] for x in image_list]))

    stitched_rows = []
    for h in unique_h_values:
        row_images = []
        for x in image_list:
            if x[0] == h:
                img = Image.open(x[2])
                row_images.append(img)

        if reverse_horizontal:
            row_images.reverse()

        row_stitched = Image.new('RGB', (sum([img.width for img in row_images]), row_images[0].height))
        x_offset = 0
        for img in row_images:
            row_stitched.paste(img, (x_offset, 0))
            x_offset += img.width
            img.close()
        stitched_rows.append(row_stitched)

    if reverse_vertical:
        stitched_rows.reverse()

    total_height = sum([img.height for img in stitched_rows])
    max_width = max([img.width for img in stitched_rows])

    final_image = Image.new('RGB', (max_width, total_height))
    y_offset = 0
    for row_img in stitched_rows:
        final_image.paste(row_img, (0, y_offset))
        y_offset += row_img.height

    return final_image

for image_id, image_list in ltr_images.items():
    stitched_ltr = stitch_images(image_list)
    extension = image_extensions[image_id]
    stitched_ltr.save(os.path.join(output_folder_path, f'{image_id}_LTR_stitched.{extension}'))

for image_id, image_list in rtl_images.items():
    stitched_rtl = stitch_images(image_list, reverse_vertical=True)
    extension = image_extensions[image_id]
    stitched_rtl.save(os.path.join(output_folder_path, f'{image_id}_RTL_stitched.{extension}'))

print("Images stitched and saved successfully!")
