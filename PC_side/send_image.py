#!/usr/bin/env python3
"""
PC-side test script: send face image(s) to STM32H747I-DISCO via USART (serial).

Protocol (single image, backward compatible):
  1. 4-byte magic: "FACE" (little-endian: 0x45434146)
  2. 4-byte image size: raw ARGB8888 bytes (little-endian uint32_t)
  3. Raw ARGB8888 pixel data: 320 x 240 x 4 = 307200 bytes

Protocol (multiple images, new):
  1. Send each image with "FACE" protocol (STM32 stores at consecutive SDRAM addrs)
  2. After last image, send: "MULT" + num_images(4B LE) + 0(4B) + no data
     STM32 then cycles through all received images.

Usage:
  python send_image.py <image_file_or_folder> [COM port] [baudrate]

Examples:
  python send_image.py face.jpg                              # Single image
  python send_image.py WIDERFace/WIDER_test/images/0--Parade # Folder
  python send_image.py . COM4                                # All images in current dir
  python send_image.py face.jpg COM3 115200                  # Full args

Requires: pip install pyserial pillow
"""

import sys
import struct
import time
import os
from pathlib import Path

try:
    import serial
except ImportError:
    print("Error: pyserial not installed. Run: pip install pyserial")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow not installed. Run: pip install pillow")
    sys.exit(1)

# ---- Configuration ----
TARGET_WIDTH    = 320
TARGET_HEIGHT   = 240
MAGIC_FACE      = b"FACE"         # Single image magic
MAGIC_MULT      = b"MULT"         # End-of-batch magic
DEFAULT_PORT    = "COM3"
BAUDRATE        = 115200
BYTES_PER_PIXEL = 4               # ARGB8888
MAX_IMAGES      = 10              # Max images STM32 can store (SDRAM: 4MB / 300KB)
IMG_EXTENSIONS  = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def find_serial_port():
    """Auto-detect STM32 serial port."""
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "STMicroelectronics" in p.description or "STLink" in p.description:
            print(f"Found STM32 port: {p.device} - {p.description}")
            return p.device
    if ports:
        print(f"No STM32 port found. Available: {[p.device for p in ports]}")
        return ports[0].device
    print("No serial ports found!")
    return None


def load_and_convert_image(image_path):
    """Load image, resize to 320x240, convert RGBA->ARGB, return raw bytes (307200)."""
    img = Image.open(image_path).convert("RGBA")
    print(f"  Original: {img.size[0]}x{img.size[1]}")
    img_resized = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.BILINEAR)

    # PIL RGBA bytes: [R, G, B, A, ...]
    # STM32 expects ARGB read as little-endian uint32: byte0=B, byte1=G, byte2=R, byte3=A
    # Swap R (byte 0) <-> B (byte 2) for each pixel
    raw = bytearray(img_resized.tobytes())
    for i in range(0, len(raw), 4):
        raw[i], raw[i + 2] = raw[i + 2], raw[i]  # R <-> B

    if len(raw) != TARGET_WIDTH * TARGET_HEIGHT * BYTES_PER_PIXEL:
        print(f"  WARNING: got {len(raw)} bytes, expected {TARGET_WIDTH * TARGET_HEIGHT * BYTES_PER_PIXEL}")
    return bytes(raw)


def collect_images(input_path):
    """Return list of image file paths from a file or folder."""
    p = Path(input_path)
    if p.is_file():
        if p.suffix.lower() in IMG_EXTENSIONS:
            return [str(p)]
        else:
            print(f"Error: '{input_path}' is not a supported image file.")
            sys.exit(1)
    elif p.is_dir():
        images = []
        for ext in IMG_EXTENSIONS:
            images.extend(sorted(p.glob(f"*{ext}")) + sorted(p.glob(f"*{ext.upper()}")))
        # Remove duplicates, sort by name
        images = sorted(set(str(img) for img in images))
        if not images:
            print(f"Error: no image files found in '{input_path}'")
            sys.exit(1)
        print(f"Found {len(images)} image(s) in folder.")
        if len(images) > MAX_IMAGES:
            print(f"Warning: truncated to {MAX_IMAGES} images (STM32 storage limit).")
            images = images[:MAX_IMAGES]
        return images
    else:
        print(f"Error: '{input_path}' is not a valid file or directory.")
        sys.exit(1)


def send_single_image(ser, raw_data, img_index, total):
    """Send one image with FACE protocol. Returns est. time in seconds."""
    total_size = len(raw_data)
    header = MAGIC_FACE + struct.pack("<I", total_size)
    payload = header + raw_data

    print(f"\n  Image {img_index}/{total}: {total_size} bytes ->", end=" ", flush=True)

    CHUNK = 4096
    sent = 0
    while sent < len(payload):
        chunk = payload[sent:sent + CHUNK]
        ser.write(chunk)
        sent += len(chunk)
    print("OK")
    return len(payload) * 10.0 / BAUDRATE


def send_mult_end(ser, num_images):
    """Send MULT end-of-batch packet. STM32 enters multi-image cycle mode."""
    header = MAGIC_MULT + struct.pack("<I", num_images) + struct.pack("<I", 0)
    ser.write(header)
    print(f"\n  Sent MULT: {num_images} image(s) total. STM32 will cycle through them.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path   = sys.argv[1]
    port_name    = sys.argv[2] if len(sys.argv) > 2 else None
    baudrate     = int(sys.argv[3]) if len(sys.argv) > 3 else BAUDRATE

    # Collect images
    image_files = collect_images(input_path)
    total = len(image_files)

    # Find serial port
    if port_name is None:
        port_name = find_serial_port()
    if port_name is None:
        print("Please specify COM port manually, e.g.: python send_image.py img.jpg COM3")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Sending {total} image(s) to {port_name} at {baudrate} baud")
    print(f"{'='*60}")

    try:
        ser = serial.Serial(port_name, baudrate, timeout=1)
        print(f"Opened {port_name}")
        time.sleep(0.5)

        for i, img_file in enumerate(image_files):
            print(f"\n[{i+1}/{total}] {img_file}")
            raw = load_and_convert_image(img_file)
            send_single_image(ser, raw, i + 1, total)
            # Wait for STM32 to process (main loop has 2s delay between frames)
            time.sleep(1.0)

        if total > 1:
            # Send MULT end-of-batch to trigger cycle mode
            send_mult_end(ser, total)
        # For single image: no MULT needed (backward compatible)

        time.sleep(0.5)

        # Read any response
        ser.timeout = 1.0
        try:
            while True:
                b = ser.read(1)
                if not b:
                    break
        except:
            pass

    except serial.SerialException as e:
        print(f"\nSerial error: {e}")
        print(f"Troubleshooting:")
        print(f"  1. Is the STM32 board connected and powered?")
        print(f"  2. Is {port_name} the correct COM port?")
        print(f"  3. Is another program using the port?")
        sys.exit(1)
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("\nSerial port closed.")

    print(f"\nFinished! Sent {total} image(s).")


if __name__ == "__main__":
    main()
