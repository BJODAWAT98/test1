import os
import random
import time
import cv2
import numpy
import torch
from basicsr.utils.download_util import load_file_from_url
from PIL import Image
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from torchvision import transforms as T
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
from ultralytics.utils import ops
import numpy as np
import sys

img_mode = 'RGBA'
global result
result = {}


def realesrgan(img, model_name="realesr-general-x4v3", denoise_strength=1, outscale=10):

    if not img:
        return

    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3,
                            num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    netscale = 4
    file_url = [
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
    ]

    # Determine model paths
    model_path = os.path.join('weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # Use dni to control the denoise strength
    dni_weight = None
    if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
        wdn_model_path = model_path.replace(
            'realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [denoise_strength, 1 - denoise_strength]

    # Restorer Class
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=10,
        half=False,
        gpu_id=None
    )

    # img = Image.open(img)
    # Convert the input PIL image to cv2 image, so that it can be processed by realesrgan
    cv_img = numpy.array(img)
    img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2BGRA)

    # Apply restoration
    try:
        output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as error:
        print('Error', error)
        # print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    else:
        # Save restored image and return it to the output Image component
        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            extension = 'png'
        else:
            extension = 'jpg'

        out_filename = f"output_{rnd_string(8)}.{extension}"
        cv2.imwrite(out_filename, output)
        global last_file
        last_file = out_filename
        return output



def rnd_string(x):
    """Returns a string of 'x' random characters
    """
    characters = "abcdefghijklmnopqrstuvwxyz_0123456789"
    result = "".join((random.choice(characters)) for i in range(x))
    return result







def store_result(imgname, num_detections, class_num, generated_text):
    if imgname not in result:
        result[imgname] = {'num_detections': num_detections, 'texts': {}}
    result[imgname]['texts'][class_num] = generated_text
    # print(generated_text)

def save_one_box(xyxy, c, det, im, gain=1.02, pad=2, BGR=False):

    if not isinstance(xyxy, torch.Tensor):  # may be list
        xyxy = torch.stack(xyxy)
    b = ops.xyxy2xywh(xyxy.view(-1, 4))  # boxes
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = ops.xywh2xyxy(b).long()
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    t1 = time.time()
    # print(det)
    # ocr for 'Pay class'    
    if c == 4:
        image = Image.fromarray(crop).convert("RGB")
        image = realesrgan(image)
        # image = Image.fromarray(crop).convert("RGB")
        # label, raw_output = parseq_model.recognize_text('parseq', image)
        # print(label)
        # print(time.time())

        image = Image.fromarray(image).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values
        generated_ids = trocr_model.generate(pixel_values)
        # print("!!!!!!!!!!!!")
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(generated_text)
        # print(time.time()-t1)
        store_result("img_name",(det),names[c],generated_text)
    # ocr for 'account number class'
    elif c == 2:
        # print(c)
        image = Image.fromarray(crop).convert("RGB")
        image = realesrgan(image)
        # image = Image.fromarray(crop).convert("RGB")
        # label, raw_output = parseq_model.recognize_text('parseq', image)
        image = Image.fromarray(image).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values
        generated_ids = trocr_model.generate(pixel_values)
        # print("!!!!!!!!!!!!")
        label = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(label,"#%^%$%$^%$^$%$^$")
        # print(time.time()-t1)
    #   img_name = p.stem
        store_result("img_name",(det),names[c],label)
    # ocr for 'date and amount class'
    elif c == 0 or c == 1:
        image = Image.fromarray(crop).convert("RGB")
        image = realesrgan(image)
        # image = Image.fromarray(image).convert("RGB")
        # label, raw_output = parseq_model.recognize_text('parseq', image)
        image = Image.fromarray(image).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values
        generated_ids = trocr_model.generate(pixel_values)
        # print("!!!!!!!!!!!!")
        label = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(label)
        # print(time.time()-t1)
    #   img_name = p.stem
        store_result("img_name",(det),names[c],label)
    
    return result
        
    

# Load a pretrained YOLOv8l model
model = YOLO('weights/best.pt')
names = model.names
model.to("cpu")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-stage1")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-stage1")
# print("ocr-model loaded")
trocr_model.to("cpu")




import concurrent.futures

def process_image(image_path):
    im = cv2.imread(image_path)
    # im = np.array(image_path)

    # Create a ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # List to store futures
        futures = []

        results = model.predict(image_path, imgsz=800, conf=0.25, max_det=5,
                                exist_ok=True, project="temp_folder")  # generator of Results objects

        # Show the results
        for r in results:
            for box in r.boxes:
                # Submit each box processing task to the executor
                future = executor.submit(save_one_box, box.xyxy, int(box.cls), len(r.boxes.cls), im)
                futures.append((future, names[int(box.cls)]))

        # Wait for all futures to complete and get the results
        for future, class_name in futures:
            crop = future.result()
            # print(class_name, int(box.cls))

    return crop







if __name__=="__main__":
    file_path_ = sys.argv[1]
    result = process_image(file_path_)
    print(result)