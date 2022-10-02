import argparse
from pathlib import Path
import cv2
import transformer
import torch
import utils
import itertools
import time

PRESERVE_COLOR = False
WIDTH = 800
HEIGHT = 480

def webcam(style_paths, width=1280, height=720, t_change=30):
    """
    Captures and saves an image, perform style transfer, and again saves the styled image.
    Reads the styled image and show in window. 
    """
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    style_paths = [Path(s) for s in style_paths]

    # Load Transformer Network
    print("Loading Transformer Networks")
    nets = [transformer.TransformerNetwork() for _ in style_paths]
    # net.load_state_dict(torch.load(style_transform_path))
    [n.load_state_dict(torch.load(s)) for s, n in zip(style_paths, nets)]
    # net = net.to(device)
    nets = [n.to(device) for n in nets]
    print("Done Loading Transformer Networks")

    # Set webcam settings
    cam = cv2.VideoCapture(-1)
    cam.set(3, width)
    cam.set(4, height)

    for net, path in itertools.cycle(zip(nets, style_paths)):
        start = time.time()
        style_img = cv2.imread(f'images/{path.stem}.jpg')
        # resized_width = width // 4
        # resized_height = int(resized_width * style_img.shape[0] / style_img.shape[1])
        style_img = cv2.resize(style_img, (180, 180))
        # cv2.imshow('Style Image', style_img)
        # Main loop
        with torch.no_grad():
            while True:
                # Get webcam input
                ret_val, img = cam.read()

                # Mirror 
                img = cv2.flip(img, 1)

                # Add image to capture
                img[height - style_img.shape[0] : height , width - style_img.shape[1] : width ] = style_img

                # Free-up unneeded cuda memory
                torch.cuda.empty_cache()
                
                # Generate image
                content_tensor = utils.itot(img).to(device)
                generated_tensor = net(content_tensor)
                generated_image = utils.ttoi(generated_tensor.detach())
                if (PRESERVE_COLOR):
                    generated_image = utils.transfer_color(img, generated_image)

                generated_image = generated_image / 255

                pressed = cv2.waitKey(1)

                # Show webcam
                cv2.imshow('Demo webcam', generated_image)
                if time.time() - start > t_change or pressed == 32: 
                    break  # esc to quit
                if pressed == 27:
                    cam.release()
                    cv2.destroyAllWindows()
                    exit(0)
    # Free-up memories
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--style", type=str, default='starry')
    args.add_argument("--t-change", type=int, default=30)
    args = args.parse_args()

    styles = ['bayanihan', 'mosaic', 'starry', 'tokyo_ghoul', 'udnie', 'wave']
    style_paths = [f"transforms/{s}.pth" for s in styles]

    STYLE_TRANSFORM_PATH = f"transforms/{args.style}.pth"
    webcam(style_paths, WIDTH, HEIGHT, args.t_change)
