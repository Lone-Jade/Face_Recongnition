#!/usr/bin/env python3
"""
PC-side test script: send face images to STM32H747I-DISCO via USART (serial).

Protocol:
  1. 4-byte magic: "FACE" (little-endian: 0x45434146)
  2. 4-byte image size: number of raw ARGB8888 bytes (little-endian uint32_t)
  3. Raw ARGB8888 pixel data: 320 × 240 × 4 = 307200 bytes

Usage:
  python send_image.py <image_file> [COM port] [baudrate]

Examples:
  python send_image.py face.jpg              # Auto-detect COM port
  python send_image.py face.jpg COM3         # Specify COM port
  python send_image.py face.jpg COM3 115200  # Full args

Requires: pip install pyserial pillow
"""

import sys
import struct
import time
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
TARGET_WIDTH   = 320
TARGET_HEIGHT  = 240
MAGIC          = b"FACE"         # 4-byte header
DEFAULT_PORT   = "COM3"          # Default serial port
BAUDRATE       = 115200
BYTES_PER_PIXEL = 4              # ARGB8888

# ---- Serial port discovery ----
def find_serial_port():
    """Try to find the STM32 serial port."""
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        # STM32 typically shows up as "STMicroelectronics STLink Virtual COM Port"
        if "STMicroelectronics" in p.description or "STLink" in p.description:
            print(f"Found STM32 port: {p.device} — {p.description}")
            return p.device
    if ports:
        print(f"No STM32 port found. Available ports: {[p.device for p in ports]}")
        return ports[0].device  # Guess first available
    else:
        print("No serial ports found!")
        return None


def load_and_convert_image(image_path):
    """
    Load an image file and convert it to ARGB8888 format at 320×240.
    Returns raw bytes of size 320*240*4 = 307200.
    """
    img = Image.open(image_path).convert("RGBA")
    print(f"Original image: {img.size[0]}×{img.size[1]}, mode={img.mode}")

    # Resize to 320×240 (fit with letterbox/padding if needed)
    img_resized = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.BILINEAR)
    print(f"Resized to: {img_resized.size[0]}×{img_resized.size[1]}")

    # Convert to raw ARGB8888 bytes
    # PIL RGBA = [R, G, B, A] per pixel in memory
    raw = img_resized.tobytes()
    print(f"Raw bytes: {len(raw)} (expected {TARGET_WIDTH * TARGET_HEIGHT * BYTES_PER_PIXEL})")

    return raw


def send_over_serial(ser, raw_data):
    """
    Send the image data to STM32 following the protocol:
      magic (4 bytes) + size (4 bytes LE uint32) + pixel data.
    """
    total_size = len(raw_data)
    header = MAGIC + struct.pack("<I", total_size)  # 8-byte header
    payload = header + raw_data
    total_payload = len(payload)

    print(f"\nSending {total_payload} bytes ({total_size} image + 8 header)...")
    print(f"  Magic:  0x{MAGIC.hex()}")
    print(f"  Size:   {total_size} (0x{total_size:08X})")
    print(f"  Baud:   {BAUDRATE}")
    est_time = total_payload * 10.0 / BAUDRATE  # 10 bits per byte (8N1)
    print(f"  Est:    {est_time:.1f} seconds...")

    # Write in chunks for progress display
    CHUNK = 4096
    sent = 0
    while sent < total_payload:
        chunk = payload[sent : sent + CHUNK]
        ser.write(chunk)
        sent += len(chunk)
        pct = sent * 100.0 / total_payload
        print(f"\r  Progress: {sent}/{total_payload} bytes ({pct:.1f}%)", end="", flush=True)

    print("\n  Done sending!")
    return est_time


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    image_path = sys.argv[1]
    port_name  = sys.argv[2] if len(sys.argv) > 2 else None
    baudrate   = int(sys.argv[3]) if len(sys.argv) > 3 else BAUDRATE

    # Validate image file
    if not Path(image_path).exists():
        print(f"Error: image file not found: {image_path}")
        sys.exit(1)

    # Find serial port
    if port_name is None:
        port_name = find_serial_port()
    if port_name is None:
        print("Please specify COM port manually, e.g.: python send_image.py img.jpg COM3")
        sys.exit(1)

    # Load and convert
    raw_data = load_and_convert_image(image_path)

    # Open serial and send
    try:
        ser = serial.Serial(port_name, baudrate, timeout=1)
        print(f"Opened {port_name} at {baudrate} baud")

        # Small delay to let STM32 stabilize
        time.sleep(0.5)

        send_over_serial(ser, raw_data)

        print("\nWaiting for STM32 to process...")
        # Optionally read any response from STM32
        ser.timeout = 2.0
        response = b""
        try:
            while True:
                b = ser.read(1)
                if not b:
                    break
                response += b
        except KeyboardInterrupt:
            pass
        if response:
            print(f"STM32 response ({len(response)} bytes): {response.decode('ascii', errors='replace')[:200]}")

    except serial.SerialException as e:
        print(f"Serial error: {e}")
        print(f"Troubleshooting:")
        print(f"  1. Is the STM32 board connected and powered?")
        print(f"  2. Is {port_name} the correct port?")
        print(f"  3. Is another program (terminal, STM32CubeProgrammer) using the port?")
        sys.exit(1)
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed.")

    print("Finished!")


if __name__ == "__main__":
    main()
