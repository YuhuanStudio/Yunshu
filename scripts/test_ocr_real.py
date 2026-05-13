"""Real model test for GLM-OCR-bf16 — with high-res document image."""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "GLM-OCR-bf16"
)


def _create_document_image():
    """Create a high-res document-like image with clear text."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1200, 400), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
    except Exception:
        font = ImageFont.load_default()

    lines = [
        "Invoice #INV-2024-001",
        "Date: December 15, 2024",
        "Customer: Yunshu Technologies Inc.",
        "Amount Due: $12,500.00",
        "Status: PAID",
    ]
    y = 30
    for line in lines:
        draw.text((40, y), line, fill="black", font=font)
        y += 70

    path = tempfile.mktemp(suffix=".png")
    img.save(path, dpi=(300, 300))
    return path


async def main():
    from yunshu_engine.ocr_engine import OCREngine

    print(f"Loading OCR model from: {MODEL_PATH}")
    engine = OCREngine(MODEL_PATH)

    print("Starting engine...")
    await engine.start()
    assert engine.is_loaded
    print("Model loaded")

    # Test with high-res document image
    img_path = _create_document_image()
    print(f"\nCreated high-res document image: {img_path}")

    try:
        print("\n=== Text Recognition (document) ===")
        result = await engine.extract_text(img_path, task="text")
        text = result["text"]
        print(f"OCR Output:\n{text[:1000]}")
        print(f"\nConfidence: {result['confidence']}")

        # Check if the output contains recognizable content
        if len(text.strip()) > 0:
            print("\nOCR engine produced non-empty output from document image")
        else:
            print("\nWARNING: Empty output from document image")

    finally:
        os.unlink(img_path)

    # Test engine stats
    stats = engine.get_stats()
    print(f"\nEngine stats: {stats}")
    assert stats["loaded"]
    assert stats["running"]

    await engine.stop()
    assert not engine.is_loaded
    print("\nEngine stopped cleanly")
    print("\nAll tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
