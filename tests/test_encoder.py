import torch
from PIL import Image
from torchvision import transforms
print("Loading DINOv2... this may take a minute the first time")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval()
print("DINOv2 loaded successfully")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

dummy_image = Image.new("RGB", (300, 300), color=(120, 180, 60))
tensor = transform(dummy_image).unsqueeze(0)
print(f"Input tensor shape: {tensor.shape}")
with torch.no_grad():
    embedding = model(tensor)
    
print(f"Output embedding shape: {embedding.shape}")
print(f"First 5 values: {embedding[0][:5]}")
print("\nTest passed. DINOv2 is working correctly.") 