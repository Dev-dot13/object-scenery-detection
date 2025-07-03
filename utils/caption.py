from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load processor and model once globally
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_caption(image_path):
    """
    Generates a descriptive caption for the given image.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise Exception(f"Error opening image: {e}")

    inputs = processor(images=image, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_length=30,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        repetition_penalty=1.2
    )
    caption = processor.decode(output[0], skip_special_tokens=True)

    print(f"[INFO] Caption: {caption}")
    return caption
