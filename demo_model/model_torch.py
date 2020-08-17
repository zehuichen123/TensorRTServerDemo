import torch
import torchvision
from volksdep.converters import torch2trt
from volksdep.calibrators import EntropyCalibrator2
from volksdep.datasets import CustomDataset

from resnet_model.ResNet import ResNet

params = torch.load('resnet_model/best_model.pth')

dummy_input = torch.ones(1, 3, 224, 224).cuda()
model = ResNet(num_classes=30, pretrained=False)
params = torch.load('resnet_model/best_model.pth')
data_param = {}
for each_key, each_param in params['state_dict'].items():
    data_param[each_key[7:]] = each_param
model.load_state_dict(data_param)
model = model.cuda().eval()
# model = torchvision.models.resnet18(pretrained=True).cuda().eval()

trt_model = torch2trt(model, dummy_input)

from volksdep.converters import save

save(trt_model, 'triton_model/plan_model/1/model.plan')