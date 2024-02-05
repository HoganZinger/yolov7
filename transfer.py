"""
@author : hogan
time:
function: transfer a pickle into a .pt
"""
import torch

# # 加载.pkl格式的模型
# loaded_model = torch.load('./pickles/neizhongpi_01_resnet50_1.pkl')
# # 保存为.pt格式的模型
# torch.save(loaded_model, 'neizhongpi.pt')

# 加载.pkl格式的模型
loaded_model = torch.load('./pickles/surface_02_resnet50_1.pkl')
# 保存为.pt格式的模型
torch.save(loaded_model, 'surface.pt')

# 加载.pkl格式的模型
loaded_model = torch.load('./pickles/side4_02_resnet50_1.pkl')
# 保存为.pt格式的模型
torch.save(loaded_model,  'side4.pt')




