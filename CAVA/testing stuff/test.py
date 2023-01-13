import multiprocessing
import os
from PIL import Image

def read_image(image_file):
    # Open the image file using PIL
    image = Image.open("colectiiImagini/"+image_file)

    # Return the image as a tuple (filename, image)
    return image_file, image

def process_images(image_files):
    # Create a pool of worker processes
    with multiprocessing.Pool() as pool:
        # Use map() to apply the read_image function to each image file
        images = pool.map(read_image, image_files)

    # Return the list of images
    return images

if __name__ == "__main__":
    # Get a list of all image files in the current directory
    image_files = [f for f in os.listdir("colectiiImagini/.") if f.endswith(".jpg")]
    print(image_files)
    # Process the images in parallel
    images = process_images(image_files)

    # Print the images
    for image_file, image in images:
        print(f"{image_file}: {image.size}")