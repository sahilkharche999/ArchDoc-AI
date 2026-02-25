import pytesseract
from PIL import Image

img = Image.open("note_page.png")
text = pytesseract.image_to_string(img, lang="eng")

print(text)
