import requests

url = 'http://127.0.0.1:5000/upload'


def get_response(url, image_path='data/test_images/15001.jpg', text='This is a sample text string.'):
    # Prepare the files and data
    with open(image_path, 'rb') as img:
        files = {'image': img}
        data = {'text': text}
        response = requests.post(url, files=files, data=data)

    # Print the response
    print(response.json())


get_response(url)
