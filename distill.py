import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO
import torch.nn.functional as F
from torchvision import transforms
from data_loader import load_dataset  # Pastikan file data_loader.py ada
from pathlib import Path

# Load teacher and student models
teacher_model = YOLO('models/yolov8m.pt')  # Teacher model (YOLOv8 medium)
student_model = YOLO('models/yolov8n.pt')  # Student model (YOLOv8 nano)

# Path dataset
data_path = 'C:/Users/phantom/kd/datasets/data.yaml'  # Path absolut ke file data.yaml Anda

# Hyperparameters
alpha = 0.5  # weight for distillation loss
temperature = 3  # temperature for softening logits
batch_size = 16
epochs = 100

# Data transformations
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize images to 640x640 (as required by YOLOv8)
    transforms.ToTensor(),
])

def distillation_loss(student_logits, teacher_logits, temperature, alpha):
    """
    Menghitung distillation loss.
    """
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
    return alpha * loss

def train_with_distillation(data_path):
    """
    Fungsi utama untuk melatih model dengan distillation.
    """
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)

    # Load dataset dari data.yaml
    student_model.train(data=data_path, epochs=epochs, batch=batch_size)

    for epoch in range(epochs):
        student_model.train()
        teacher_model.eval()  # Freeze teacher model during training

        running_loss = 0.0

        # Here you should get your dataloader and batch processing (if necessary)
        train_loader, val_loader = load_dataset(data_path, batch_size, transform)

        for batch in train_loader:
            images, _ = batch  # Dataloader returns images and labels, we only need images

            # Move data to the same device as the model
            images = images.to(student_model.device)

            # Get predictions from both models
            with torch.no_grad():
                teacher_out = teacher_model(images)
            student_out = student_model(images)

            # Calculate distillation loss
            distill_loss = distillation_loss(student_out['pred'], teacher_out['pred'], temperature, alpha)
            
            # Combine with student model's original loss
            student_loss = student_out['loss']
            total_loss = student_loss + distill_loss

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}')

    # Save student model after distillation
    student_model.save('results/student_model_distilled.pt')

if __name__ == "__main__":
    # Pastikan file data.yaml sudah diatur dengan benar
    train_with_distillation(data_path)
