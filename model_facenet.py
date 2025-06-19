import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
device = "cuda:0" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose(
    [
        transforms.Resize(size=(224,224)),
        transforms.ToTensor()
    ]
)
model = InceptionResnetV1(pretrained='vggface2').to(device=device).eval()
def dot_product(path,captured_photo):
    img1 = Image.open(path)
    img1 = transform(img1).unsqueeze(0).to(device=device)
    img1_embed = model(img1).squeeze()
    img2_embed = captured_photo
    dot_product = torch.dot(img1_embed,img2_embed)
    mag1 = torch.sum(img1_embed ** 2) ** 0.5
    mag2 = torch.sum(img2_embed ** 2) ** 0.5
    return dot_product / (mag1*mag2)
