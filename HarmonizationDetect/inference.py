import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
import albumentations
import HarmonizationDetect.utils.PPNetBulider

import cv2
import numpy as np

model = HarmonizationDetect.utils.PPNetBulider.PPNet('resnet50', False, 2)
model = model.cpu()
state = torch.load('pretrained/bestModel.pth')
# print("test:")
model.load_state_dict(state)
model.eval()


# Calculate the harmonization scores. Please refer to Section 3.3 in our paper-v2 for details.
def harmon_detect(img_path, mask_path):
    transform_mean = [.5, .5, .5]
    transform_var = [.5, .5, .5]
    torch_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(transform_mean, transform_var)])

    real_image = cv2.imread(img_path)
    real_image = albumentations.Resize(256, 256)(image=cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB))
    real_image = torch_transform(real_image['image'])

    mask = cv2.imread(mask_path)
    mask = albumentations.Resize(256, 256)(image=mask[:, :, 0].astype(np.float32) / 255.)
    mask = transforms.ToTensor()(mask['image'])

    real_image = real_image.unsqueeze(0).cpu()
    mask = mask.unsqueeze(0).cpu()

    with torch.no_grad():
        output = model(real_image, mask)
        output = F.softmax(output, dim=1)

        # return possibility
        return output[0][1].item()