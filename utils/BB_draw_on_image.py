from PIL import Image, ImageDraw, ImageFont
import os

def draw_boxes_on_image(image_path, data_list, output_path="debug_boxes.png"):
    """
    Draws bounding boxes from a FLAT LIST of dictionaries.
    Format: [{'title': '...', 'coords': {'x1':...}}, ...]
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    colors = ["red", "blue", "green", "orange", "purple", "magenta"]
    
    print(f"Drawing boxes on {image_path}...")

    # --- UPDATED LOOP FOR LIST INPUT ---
    for i, item in enumerate(data_list):
        title = item["title"]
        c = item["coords"]
        
        color = colors[i % len(colors)]
        
        # Coordinates
        x1, y1, x2, y2 = c["x1"], c["y1"], c["x2"], c["y2"]
        
        # Draw Box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=8)
        
        # Draw Label
        text_label = title
        bbox = draw.textbbox((x1, y1), text_label, font=font)
        draw.rectangle([bbox[0], bbox[1]-10, bbox[2]+10, bbox[3]+10], fill=color)
        draw.text((x1+5, y1-5), text_label, fill="white", font=font)
        
        print(f"  - Drew {title} at [{x1}, {y1}, {x2}, {y2}]")

    img.save(output_path)
    print(f"Saved debug image to: {output_path}")

# --- RUN IT ---
if __name__ == "__main__":
    img_path = "output_temp/floor_1.png"
    
    # PASTE YOUR LIST HERE (Not the dictionary)
    flat_data = [{'title': 'FOUNDATION PLAN LEGEND', 'coords': {'x1': 1220, 'y1': 6029, 'x2': 1952, 'y2': 6081}, 'area': 38064}, {'title': 'SHEAR WALL SCHEDULE', 'coords': {'x1': 2501, 'y1': 4716, 'x2': 3148, 'y2': 4768}, 'area': 33644}, {'title': 'BASE PLATE SCHEDULE', 'coords': {'x1': 2560, 'y1': 3305, 'x2': 3189, 'y2': 3357}, 'area': 32708}, {'title': 'FOOTING SCHEDULE', 'coords': {'x1': 2618, 'y1': 4135, 'x2': 3161, 'y2': 4189}, 'area': 29322}, {'title': 'FOUNDATION PLAN', 'coords': {'x1': 5079, 'y1': 6677, 'x2': 6486, 'y2': 6781}, 'area': 146328}]
    draw_boxes_on_image(img_path, flat_data)

