import torch
from torchvision import transforms
from PIL import Image
from main_final import MyArchitecture  

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()  
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    input_data = transform(image).unsqueeze(0)  
    return input_data

def infer(model, input_data):
    with torch.no_grad():
        output = model(input_data)
    return output

def get_predicted_class(output):
    _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

if __name__ == "__main__":
    model_path = 'trained_model.pt'  
    image_path = 'Dataset/train/15_jpg.rf.c973d4e030dcb73593251dbd281abb94.jpg' 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MyArchitecture(num_classes=3).to(device)
    model = load_model(model, model_path)

    input_data = preprocess_image(image_path).to(device)
    output = infer(model, input_data)

    predicted_class = get_predicted_class(output)

    print("Predicted Class:", predicted_class)
    print("Inference Result:", output)
