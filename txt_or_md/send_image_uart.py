#!/usr/bin/env python3
"""
PC-side test script: send face images to STM32H747I-DISCO via UART.

Protocol (simple binary):
  4 bytes  : image width  (uint32, little-endian)
  4 bytes  : image height (uint32, little-endian)
  N bytes  : RGB888 pixel data (row-major, top-to-bottom)

Usage:
  python send_image_uart.py <image_file> [--port COM3] [--baud 115200]

Requirements:
  pip install pyserial pillow
"""

import argparse
import struct
import sys
import time
import serial
from PIL import Image


def send_image(ser, img: Image.Image):
    """Send image to STM32 board over serial."""
    # Ensure RGB format
    if img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size
    pixels = img.tobytes()

    print(f"Sending image: {width}x{height}, {len(pixels)} bytes")

    # Send header: width (4 bytes LE) + height (4 bytes LE)
    header = struct.pack("<II", width, height)
    ser.write(header)
    ser.flush()
    time.sleep(0.01)  # small delay for STM32 to parse header

    # Send pixel data in chunks (to avoid overwhelming the UART buffer)
    CHUNK_SIZE = 1024
    total_sent = 0
    for i in range(0, len(pixels), CHUNK_SIZE):
        chunk = pixels[i : i + CHUNK_SIZE]
        ser.write(chunk)
        total_sent += len(chunk)
        ser.flush()
        time.sleep(0.001)  # 1ms between chunks

    print(f"Sent {total_sent} bytes. Waiting for detection result...")
    # Wait for response (detection count + coordinates)
    # The board sends back UART text: e.g. "DET:2\n" then box coordinates
    time.sleep(3.0)
    response = b""
    while ser.in_waiting > 0:
        response += ser.read(ser.in_waiting)
        time.sleep(0.1)
    if response:
        print(f"Board response:\n{response.decode('utf-8', errors='replace')}")


def main():
    parser = argparse.ArgumentParser(
        description="Send face image to STM32H747I-DISCO over UART"
    )
    parser.add_argument("image", help="Path to input image file (jpg/png/bmp)")
    parser.add_argument(
        "--port", default="COM3", help="Serial port (default: COM3)"
    )
    parser.add_argument(
        "--baud", type=int, default=115200, help="Baud rate (default: 115200)"
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[320, 320],
        metavar=("W", "H"),
        help="Resize image before sending (default: 320 320)",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Send at original resolution (up to 320x320)",
    )
    args = parser.parse_args()

    # Load and preprocess image
    img = Image.open(args.image)
    print(f"Loaded: {args.image}, original size: {img.size}")

    if not args.no_resize:
        target_w, target_h = args.resize
        # Preserve aspect ratio, pad with black
        img.thumbnail((target_w, target_h), Image.LANCZOS)
        # Create a black canvas of target size and paste centered
        canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        paste_x = (target_w - img.width) // 2
        paste_y = (target_h - img.height) // 2
        canvas.paste(img, (paste_x, paste_y))
        img = canvas
        print(f"Resized to: {img.size}")

    # Open serial port
    try:
        ser = serial.Serial(
            port=args.port,
            baudrate=args.baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=5.0,
        )
        print(f"Opened {args.port} @ {args.baud} baud")
    except serial.SerialException as e:
        print(f"ERROR: Cannot open {args.port}: {e}")
        print("Available ports:")
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}: {p.description}")
        sys.exit(1)

    try:
        send_image(ser, img)
    finally:
        ser.close()
        print("Serial port closed.")


if __name__ == "__main__":
    main()
