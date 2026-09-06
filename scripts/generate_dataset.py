"""
Generates 120 synthetic benchmark images (256x256 RGB PPM format)
simulating SIPI/MNIST synthetic textures with geometric shapes, gradients,
and edge features to evaluate the CUDA pipeline on over 100 images.
"""

import os
import math

def generate_image(idx, width=256, height=256):
    pixels = bytearray()
    center_x = width // 2
    center_y = height // 2
    radius = 30 + (idx % 40)
    
    for y in range(height):
        for x in range(width):
            # Distance to center
            dx = x - center_x
            dy = y - center_y
            dist = math.sqrt(dx*dx + dy*dy)
            
            # Circle edge
            if abs(dist - radius) < 4:
                r, g, b = 255, 255, 255
            elif dist < radius:
                r = (x * 2 + idx * 5) % 256
                g = (y * 2 + idx * 7) % 256
                b = (idx * 11) % 256
            else:
                # Diagonal grid and background gradient
                if (x + y) % (16 + (idx % 16)) == 0:
                    r, g, b = 200, 200, 200
                else:
                    r = (x + idx * 3) % 180
                    g = (y + idx * 2) % 180
                    b = 40
            pixels.extend([r, g, b])
            
    return bytes(pixels)

def main():
    out_dir = "data/input"
    os.makedirs(out_dir, exist_ok=True)
    num_images = 120
    print(f"Generating {num_images} PPM benchmark images in {out_dir}...")
    
    for i in range(num_images):
        filename = f"img_{i:03d}.ppm"
        filepath = os.path.join(out_dir, filename)
        raw_data = generate_image(i, 256, 256)
        with open(filepath, "wb") as f:
            header = f"P6\n256 256\n255\n".encode("ascii")
            f.write(header)
            f.write(raw_data)
            
    print(f"Successfully generated {num_images} images (256x256 RGB)!")

if __name__ == "__main__":
    main()
