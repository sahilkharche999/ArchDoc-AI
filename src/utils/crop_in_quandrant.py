#crop_in_quandrant.py
from PIL import Image
import cv2
import os
def get_height_and_width(image_path:str):
    if not os.path.exists(image_path):
        raise ValueError("Image does not exist")
    try:
       with Image.open(image_path) as img:
           width,height=img.size
           return width,height
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
        return None, None
    except Exception as e:
        print(f"An erro occured : {e}")
        return None,None


# def crop_image_into_quad(image_file,output_path):
#     width,height=get_height_and_width(image_file)
#     if not width or not height:
#       raise ValueError("Image does not exist")
#     if not os.path.exists(output_path):
#         os.mkdir(output_path)
#     mid_x = width  // 2
#     mid_y = height // 2
#     try: 
#         img = cv2.imread(image_file)
#         top_left_img=img[0:mid_y,0:mid_x]
#         top_right_img=img[0:mid_y,mid_x:width]
#         bottom_left_img=img[mid_y:height,0:mid_x]
#         bottom_right_img=img[mid_y:height,mid_x:width]

#         cv2.imwrite(f"{output_path}/top_left_img.png", top_left_img)
#         cv2.imwrite(f"{output_path}/top_right_img.png", top_right_img)
#         cv2.imwrite(f"{output_path}/bottom_left_img.png", bottom_left_img)
#         cv2.imwrite(f"{output_path}/bottom_right_img.png", bottom_right_img)
#     except Exception as e:
#         raise ValueError(f"Got issue as {e}")
#     return True


def crop_image_into_quad(image_file, output_path, overlap_ratio=0.12, zoom_factor=2):
    # Load image
    img = cv2.imread(image_file)
    if img is None:
        raise ValueError("Failed to read image")

    height, width = img.shape[:2]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    rows = 2
    cols = 4

    tile_w = width / cols
    tile_h = height / rows

    pad_x = int(tile_w * overlap_ratio)
    pad_y = int(tile_h * overlap_ratio)

    for r in range(rows):
        for c in range(cols):
            x1 = int(c * tile_w)
            y1 = int(r * tile_h)
            x2 = int((c + 1) * tile_w)
            y2 = int((r + 1) * tile_h)

            # Apply overlap padding
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_y)

            tile_img = img[y1:y2, x1:x2]

            # Zoom
            tile_img = cv2.resize(
                tile_img,
                None,
                fx=zoom_factor,
                fy=zoom_factor,
                interpolation=cv2.INTER_CUBIC
            )

            # Row-wise, column-wise naming
            row_name = "top" if r == 0 else "bottom"
            col_names = ["left", "second", "third", "last"]
            filename = f"{output_path}/{row_name}_{col_names[c]}.png"

            cv2.imwrite(filename, tile_img)

    return True


# image_file='Floor.png'
# output_path='crop_in_quadrants'
# if crop_image_into_quad(image_file,output_path):
#     print(f'Crop imaged saved in the folder : {output_path} sucessfully')
# else :
#     print('something went wrong')




