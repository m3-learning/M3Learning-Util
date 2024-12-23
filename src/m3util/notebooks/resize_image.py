import sys
import argparse
from PIL import Image


def resize_image_by_height(image_path, target_height=400, output_path=None):
    """
    Resizes an image to a specified height, preserving aspect ratio.
    If the image is smaller than the target height, no resizing is performed.

    :param image_path: Path to the input image file.
    :param target_height: Desired height in pixels (default: 400).
    :param output_path: Path to save the resized image. If None, image is replaced in place.
    """
    # Open the image
    img = Image.open(image_path)
    width, height = img.size

    # Check if the image needs to be resized
    if height > target_height:
        # Calculate scale factor
        scale_factor = target_height / float(height)
        # Compute new width
        new_width = int(width * scale_factor)

        # Resize the image
        resized_img = img.resize((new_width, target_height))

        # Determine output path
        if output_path is None:
            output_path = image_path

        # Save the resized image
        resized_img.save(output_path)
    else:
        # Image is already small enough, optionally copy if output_path is provided
        if output_path and output_path != image_path:
            img.save(output_path)
        # If no output_path is given and image is small, do nothing


def main(args=None):
    """
    Entry point for the image resizer.
    """
    parser = argparse.ArgumentParser(
        description="Resize an image by height while preserving aspect ratio."
    )
    parser.add_argument("image_path", help="Path to the image to resize.")
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        default=400,
        help="Target height in pixels (default: 400).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path for the resized image. If not provided, original image is replaced.",
    )

    if args is None:
        args = sys.argv[1:]

    parsed_args = parser.parse_args(args)

    resize_image_by_height(
        parsed_args.image_path,
        target_height=parsed_args.height,
        output_path=parsed_args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
