import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def draw_bboxes(
    content_list_path: str,
    page_image_path: str,
    output_image_path: str,
    skip_discarded: bool = True
):
    # Load content list
    with open(content_list_path, "r", encoding="utf-8") as f:
        content_list = json.load(f)

    # Load page image
    img = Image.open(page_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    img_w, img_h = img.size

    print(f"Image size: {img_w} x {img_h}")

    try:
        font = ImageFont.load_default()
    except:
        font = None

    for idx, item in enumerate(content_list):

        if "bbox" not in item:
            continue

        if skip_discarded and item.get("type") == "discarded":
            continue

        mx0, my0, mx1, my1 = item["bbox"]

        # 🔥 Correct scaling (0–1000 → pixel space)
        x0 = int((mx0 / 1000) * img_w)
        x1 = int((mx1 / 1000) * img_w)
        y0 = int((my0 / 1000) * img_h)
        y1 = int((my1 / 1000) * img_h)

        comp_type = item.get("type", "unknown")

        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)

        label = f"{idx}:{comp_type}"
        draw.text((x0 + 2, y0 + 2), label, fill="red", font=font)

    output_image_path = Path(output_image_path)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_image_path)

    print(f"✅ Saved: {output_image_path}")

draw_bboxes('output/floor_plan/auto/floor_plan_content_list.json','utils/floor_1.png','output/floor_plan/debug_bbox_overlay.png',)