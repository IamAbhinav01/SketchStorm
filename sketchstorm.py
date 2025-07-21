from mcp.server.fastmcp import FastMCP
from diffusers import StableDiffusionPipeline
import torch, threading, time
import cv2, numpy as np
from PIL import Image
import tkinter as tk

mcp = FastMCP("ClaudeSketcher")
SAVE_PATH = "generated.png"

# Setup optimized model loading
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe.to(device)
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
pipe.enable_attention_slicing()
_ = pipe("warmup")  # Warm up to load weights in advance

@mcp.tool()
def sketch_from_prompt(prompt: str) -> str:
    """
    Accepts prompt, starts image generation and live sketching in background.
    Returns immediately to avoid Claude timeout.
    """
    threading.Thread(target=process_and_draw, args=(prompt,), daemon=True).start()
    return f"🎨 Drawing '{prompt}'... Sketching will appear live on screen!"

def process_and_draw(prompt):
    image = pipe(prompt).images[0]
    image.save(SAVE_PATH)
    draw_contours(SAVE_PATH)

def draw_contours(path):
    image = cv2.imread(path, 0)
    edges = cv2.Canny(image, 80, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    root = tk.Tk()
    root.title("Claude AI Sketcher")
    canvas = tk.Canvas(root, width=512, height=512, bg='white')
    canvas.pack()

    def draw():
        for contour in contours:
            for i in range(1, len(contour)):
                x1, y1 = contour[i - 1][0]
                x2, y2 = contour[i][0]
                canvas.create_line(x1, y1, x2, y2, fill='black')
                root.update()
                time.sleep(0.001)

    threading.Thread(target=draw).start()
    root.mainloop()

if __name__ == "__main__":
    mcp.run()
