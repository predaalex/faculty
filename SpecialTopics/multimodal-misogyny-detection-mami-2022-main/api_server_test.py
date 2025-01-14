import requests
import base64

url = 'http://127.0.0.1:5000/upload'


def get_response(url, image_path='data/test_images/15001.jpg', text='This is a sample text string.'):
    # Encode the image as base64
    with open(image_path, 'rb') as img:
        base64_image = base64.b64encode(img.read()).decode('utf-8')
        # Prepare the data
        data = {
            'text': text,
            'image': base64_image
        }

        # Send the POST request
        response = requests.post(url,  data=data)

        # Print the response
        print(response.json())


get_response(url)
