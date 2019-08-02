import io
import json
from PIL import Image
import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb
from network import resnet34
from utils import Tester


def preprocess_image(img):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        # normalize
    ])

    img_pil = Image.open(img)
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    img_input = img_variable.cuda()
    return img_input

def returnCAM(feature_conv, weight_softmax):
    # generate the class activation maps upsample to 256x256
    #img_pil = Image.open(img)
    #w, h = img_pil.size
    size_upsample = (96, 96)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    cam = weight_softmax.dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

#Parameters:
model_path='./models/model_34/ckpt_epoch_60.pth'
finalconv_name = 'layer4'
features_blobs = []
image_path='test_cloth.jpg'
# load model:
model = resnet34(pretrained=False)
#model=torch.nn.DataParallel(model)
model_state = torch.load(model_path)
t_state = {}
for k, v in model_state.items():
    t_state[k[7:]]=v

model.load_state_dict(t_state)
print('Load ckpt from',model_path,'\n')
model = model.cuda()
model.eval()


#hook
handle=model._modules.get(finalconv_name).register_forward_hook(hook_feature)

# feed the image and prediction:
img_input=preprocess_image(image_path)

output = model(img_input)
#print(output)


score_hat = F.softmax(output[0], dim=1)
score_cloth = F.softmax(output[1], dim=1)
_, prediction_hat = torch.max(score_hat.data, dim=1)
_, prediction_cloth = torch.max(score_cloth.data, dim=1)

result_label_hat_np = prediction_hat.data.cpu().numpy()[0]
prob_hat=score_hat.squeeze().data[result_label_hat_np]

result_label_cloth_np = prediction_cloth.data.cpu().numpy()[0]
prob_cloth=score_cloth.squeeze().data[result_label_cloth_np]

print('Label of hat:',result_label_hat_np,'Prob of hat:',prob_hat,'\n',
      'Label of cloth:', result_label_cloth_np,'Prob of cloth:',prob_cloth,'\n')

# hook the feature extractor

#print(model)
#model=
# get the softmax weight
print(handle)
params = list(model.parameters())
#print(params)
weight_softmax_hat = np.squeeze(params[-4].data.cpu().numpy()[0])
weight_softmax_cloth = np.squeeze(params[-2].data.cpu().numpy()[1])

# Generate class activation mapping for the hat prediction:
CAMs = returnCAM(features_blobs[0], weight_softmax_cloth)

img = cv2.imread(image_path)
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('CAM.jpg', result)
