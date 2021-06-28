import io
import base64
import requests

import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, request, redirect
from flask_ngrok import run_with_ngrok

from flask_bootstrap import Bootstrap

app = Flask(__name__)
run_with_ngrok(app)
Bootstrap(app)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()


def transform_image(image):
    tforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(
                                     [0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
    return tforms(image).unsqueeze(0) / 255.0


def get_prediction(image):
    tensor = transform_image(image=image)
    outputs = model.forward(tensor)
    return outputs


def plot_preds(source_img, boxs, lbels):
    labels = lbels['labels'].detach().numpy()
    boxes = boxs['boxes'].detach().numpy()

    #список классов модели(MS COCO 2017)
    list_classes = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
    draw = ImageDraw.Draw(source_img)
    font = ImageFont.truetype(r'/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf', 20)
    for id, box in enumerate(boxes):
        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline='red')
        draw.text((box[0], box[1]), list_classes[int(labels[id]-1)])
    return source_img


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        url = request.form.get('url')
        try:
            r = requests.get(url)
        except requests.exceptions.RequestException:
            return redirect(request.url)
        image_bytes = r.content
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        preds = get_prediction(image=image)

        #устанавливаем порог score'а
        THRESH = 0.5

        #вытаскиваем предсказанные данные из outputs
        boxes = preds[0]['boxes'][preds[0]['scores'] > THRESH]
        labels = preds[0]['labels'][preds[0]['scores'] > THRESH]
        labels_dict = dict()
        labels_dict['labels'] = labels
        boxes_dict = dict()
        boxes_dict['boxes'] = boxes

        img_with_boxes = plot_preds(image, boxes_dict, labels_dict)
        output = io.BytesIO()
        img_with_boxes.save(output, format='PNG')
        output.seek(0, 0)
        output_s = output.read()
        b64 = base64.b64encode(output_s).decode('ascii')
        return render_template('result.html', img_original=url, img_with_boxes=b64)
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
