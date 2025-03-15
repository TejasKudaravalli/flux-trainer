from io import BytesIO
import zipfile
from PIL import Image
import os


def scale_down_image(image, max_size=1024):
    image.thumbnail((max_size, max_size))  # Resize while maintaining aspect ratio
    img_buffer = BytesIO()
    image.save(img_buffer, format="JPEG")  # Use JPEG to optimize size
    img_buffer.seek(0)  # Reset buffer position
    return img_buffer


def resize_zip_file(uploaded_zip_file):
    MAX_FILE_SIZE = 100 * 1024 * 1024
    uploaded_zip_file = BytesIO(uploaded_zip_file.read())
    uploaded_zip_file.seek(0, os.SEEK_END)
    zip_size = uploaded_zip_file.tell()
    if zip_size > MAX_FILE_SIZE:
        output_buffer = BytesIO()
        with (
            zipfile.ZipFile(uploaded_zip_file, "r") as archive,
            zipfile.ZipFile(output_buffer, "w", zipfile.ZIP_DEFLATED) as output_zip,
        ):
            for file_name in archive.namelist():
                with archive.open(file_name) as file:
                    if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        image = Image.open(file).convert("RGB")
                        resized_img = scale_down_image(image)
                        output_zip.writestr(file_name, resized_img.getvalue())
        output_buffer.seek(0)
        return output_buffer
    else:
        uploaded_zip_file.seek(0)
        return uploaded_zip_file
