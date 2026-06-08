#!/usr/bin/env python3
"""
Convert any image (jpg/png/bmp) to ARGB8888 C header for STM32 LCD display.

Outputs a .h file with a const uint32_t array suitable for the baseline
STM32H747I-DISCO project's Images[] mechanism.

Usage:
  python convert_image_to_c.py <input_image> [--name NAME] [--size W H]
"""
import argparse
import sys

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Convert image to ARGB8888 C header")
    parser.add_argument("input", help="Input image file (jpg/png/bmp)")
    parser.add_argument("--name", default=None,
                        help="Variable name (default: derived from filename)")
    parser.add_argument("--size", type=int, nargs=2, metavar=("W", "H"),
                        default=[320, 240], help="Target size (default: 320 240)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output .h file (default: <name>.h)")
    args = parser.parse_args()

    img = Image.open(args.input).convert("RGBA")
    target_w, target_h = args.size

    # Resize preserving aspect ratio, pad with black
    img.thumbnail((target_w, target_h), Image.LANCZOS)
    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 255))
    paste_x = (target_w - img.width) // 2
    paste_y = (target_h - img.height) // 2
    canvas.paste(img, (paste_x, paste_y))
    img = canvas

    name = args.name or "face_img"
    out_path = args.output or f"{name}.h"
    guard = f"__{name.upper()}_H__"

    pixels = list(img.getdata())  # list of (R, G, B, A)
    total = target_w * target_h

    with open(out_path, "w") as f:
        f.write("/**\n")
        f.write(f"  * @brief   ARGB8888 image {target_w}x{target_h}\n")
        f.write(f"  * @source  {args.input}\n")
        f.write("  */\n")
        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"const uint32_t {name}[{total}] =\n")
        f.write("{\n")

        for i, (r, g, b, a) in enumerate(pixels):
            val = (a << 24) | (r << 16) | (g << 8) | b
            # Format as 0xAARRGGBB hex
            line_end = ""
            if i < total - 1:
                comma = ","
                if (i + 1) % 8 == 0:
                    line_end = "\n"
                else:
                    line_end = " "
            else:
                comma = ""
                line_end = "\n"
            f.write(f"0x{val:08X}{comma}{line_end}")

        f.write("};\n\n")
        f.write(f"#endif /* {guard} */\n")

    print(f"Generated: {out_path}")
    print(f"  Size: {target_w}x{target_h}, {total} pixels")
    print(f"  Size on disk: ~{total * 4 / 1024:.0f} KB")


if __name__ == "__main__":
    main()
