from flask import Flask, request, jsonify
from PIL import Image
import io
import torch.optim as optim
import time
from sklearn import metrics
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from helper_functions import *
from text_normalizer import *
import argparse
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
from train_multitask import MMNetwork, tokenize

_tokenizer = _Tokenizer()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_model, _ = clip.load("ViT-L/14", jit=False)
input_resolution = clip_model.visual.input_resolution
clip_model.float().eval()
clip_model = nn.DataParallel(clip_model)

model = MMNetwork(768, 768, 1)
model.to(device)

img_transformer = transforms.Compose([
        transforms.Resize((input_resolution, input_resolution), interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.CenterCrop(clip_model.visual.input_resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])

app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload_image_and_text():
    try:
        # Check if the request contains a file
        if 'image' not in request.files:
            return jsonify({'error': 'No image part in the request'}), 400

        file = request.files['image']

        # Check if a file was actually selected
        if file.filename == '':
            return jsonify({'error': 'No selected image'}), 400

        # Check if the request contains a text string
        if 'text' not in request.form:
            return jsonify({'error': 'No text part in the request'}), 400

        text = request.form['text']  # Retrieve the text string
        text_tokens, masks = tokenize(text)
        text_tokens = text_tokens.unsqueeze(0).to(device)
        masks = masks.unsqueeze(0).to(device)

        image = Image.open(file.stream).convert('RGB')
        img_input = img_transformer(image).unsqueeze(0).to(device)

        # Generate model response.
        clip_model.to(device)
        model.to(device)

        img_feats = clip_model.module.encode_image(img_input)
        txt_feats = clip_model.module.encode_text(text_tokens)
        outputs1, outputs2, outputs3, outputs4, outputs5 = model(img_feats, txt_feats, masks)

        preds1 = (outputs1 > 0.5).int()
        preds2 = (outputs2 > 0.5).int()
        preds3 = (outputs3 > 0.5).int()
        preds4 = (outputs4 > 0.5).int()
        preds5 = (outputs5 > 0.5).int()

        # Return a response
        response = {
            'message': 'Image and text received successfully',
            'misogynous': preds1.item(),
            'shaming': preds2.item(),
            'stereotype': preds3.item(),
            'objectification': preds4.item(),
            'violence': preds5.item(),
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':

    app.run(debug=True)