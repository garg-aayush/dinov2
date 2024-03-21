from dinov2.models.vision_transformer import vit_base
import torch
from torchvision import transforms
from PIL import Image
import requests


# model
model = vit_base(img_size=518, patch_size=14, init_values=1.0, ffn_layer="mlp", block_chunks=0)

# # print the model parameters
# for name, param in model.named_parameters():
#     print(name, param.size())

# load the model weights
state_dict = torch.hub.load_state_dict_from_url(
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    map_location="cpu")

# load the model weights
model.load_state_dict(state_dict)


# image
# two cat images
url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba" 
image = Image.open(requests.get(url, stream=True).raw)

# preprocess the image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

preprocessed_image = transform(image).unsqueeze(0)

# make the prediction
output_features = model.forward_features(preprocessed_image)
for k,v in output_features.items():
    if isinstance(v, torch.Tensor): print(k, v.shape)
    else: print(k, v)