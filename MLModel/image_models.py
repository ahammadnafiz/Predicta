# MLModel/image_training.py
import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from torchvision.models import resnet18, efficientnet_b0, vgg16, resnet50

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = os.listdir(data_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
class ImageModel:
    def __init__(self, image_data):
        self.image = image_data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.sidebar.markdown('----')
        st.sidebar.markdown("## Image Processing Options")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image, preprocessing_options):
        """Apply selected preprocessing techniques to the image"""
        if preprocessing_options.get('resize', False):
            size = preprocessing_options.get('resize_dimensions', (224, 224))
            image = image.resize(size)
        
        if preprocessing_options.get('grayscale', False):
            image = image.convert('L')
        
        if preprocessing_options.get('normalize', False):
            image = self.transform(image)
        
        return image

    def image_classification(self, model_name='resnet50', preprocessing_options=None):
        """Perform image classification using selected model"""
        if preprocessing_options is None:
            preprocessing_options = {'resize': True, 'normalize': True}

        try:
            # Load and preprocess image
            image = Image.open(self.image)
            processed_image = self.preprocess_image(image, preprocessing_options)
            
            # Select model
            if model_name == 'resnet50':
                model = resnet50(pretrained=True)
            elif model_name == 'efficientnet':
                model = efficientnet_b0(pretrained=True)
            elif model_name == 'vgg16':
                model = vgg16(pretrained=True)
            else:
                st.error("Invalid model selection")
                return
            
            model = model.to(self.device)
            model.eval()
            
            # Make prediction
            with torch.no_grad():
                input_tensor = processed_image.unsqueeze(0).to(self.device)
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Display results
            st.subheader("Classification Results")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Show top 5 predictions
            _, indices = torch.sort(probabilities, descending=True)
            st.write("Top 5 predictions:")
            for idx in indices[:5]:
                st.write(f"{idx}: {probabilities[idx].item():.2%}")
                
        except Exception as e:
            st.error(f"Error in classification: {str(e)}")

    def object_detection(self, model_name='yolov5s', confidence_threshold=0.5):
        """Perform object detection using selected model"""
        try:
            # Load YOLOv5 model
            model = torch.hub.load('ultralytics/yolov5', model_name)
            model.conf = confidence_threshold
            
            # Load and process image
            image = Image.open(self.image)
            
            # Run inference
            results = model(image)
            
            # Display results
            st.subheader("Object Detection Results")
            st.image(results.render()[0], caption="Detection Results", use_column_width=True)
            
            # Show detection details
            detections = results.pandas().xyxy[0]
            st.write("Detected Objects:")
            st.dataframe(detections)
            
        except Exception as e:
            st.error(f"Error in object detection: {str(e)}")

    def image_segmentation(self, model_name='deeplabv3_resnet50'):
        """Perform image segmentation using selected model"""
        try:
            # Load segmentation model
            model = torch.hub.load('pytorch/vision', model_name, pretrained=True)
            model.eval().to(self.device)
            
            # Load and preprocess image
            image = Image.open(self.image)
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = model(input_tensor)['out'][0]
                output_predictions = output.argmax(0).cpu().numpy()
            
            # Display results
            st.subheader("Segmentation Results")
            st.image(image, caption="Original Image", use_column_width=True)
            st.image(output_predictions, caption="Segmentation Map", use_column_width=True)
            
        except Exception as e:
            st.error(f"Error in segmentation: {str(e)}")

class ImageTrainer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def preprocess_dataset(self, img_size=224, augmentation=False):
        """Set up data preprocessing and augmentation"""
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ]
        
        if augmentation:
            train_transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ] + transform_list
            
            train_transform = transforms.Compose(train_transform_list)
        else:
            train_transform = transforms.Compose(transform_list)
            
        val_transform = transforms.Compose(transform_list)
        
        return train_transform, val_transform

    def prepare_data(self, train_transform, val_transform, val_split=0.2, batch_size=32):
        """Prepare train and validation datasets"""
        full_dataset = ImageDataset(self.data_dir, transform=train_transform)
        
        train_size = int((1 - val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size])
        
        val_dataset.dataset.transform = val_transform
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=2)
        
        return train_loader, val_loader, len(full_dataset.classes)

    def train_model(self, model_name='resnet18', epochs=10, learning_rate=0.001, 
                   batch_size=32, img_size=224, augmentation=True):
        """Train the selected model"""
        train_transform, val_transform = self.preprocess_dataset(
            img_size=img_size, augmentation=augmentation)
        
        train_loader, val_loader, num_classes = self.prepare_data(
            train_transform, val_transform, batch_size=batch_size)
        
        # Model selection
        if model_name == 'resnet18':
            model = resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'efficientnet':
            model = efficientnet_b0(pretrained=True)
            model._fc = nn.Linear(model._fc.in_features, num_classes)
            
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training progress bars
        epoch_bar = st.progress(0)
        batch_bar = st.progress(0)
        metrics_placeholder = st.empty()
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                batch_bar.progress((i + 1) / len(train_loader))
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            train_acc = 100. * correct / total
            val_acc = 100. * val_correct / val_total
            
            metrics_placeholder.write(f"Epoch [{epoch+1}/{epochs}]")
            metrics_placeholder.write(f"Train Loss: {running_loss/len(train_loader):.3f}")
            metrics_placeholder.write(f"Train Accuracy: {train_acc:.2f}%")
            metrics_placeholder.write(f"Validation Accuracy: {val_acc:.2f}%")
            
            epoch_bar.progress((epoch + 1) / epochs)
            
            # if val_acc > best_val_acc:
            #     best_val_acc = val_acc
            #     torch.save(model.state_dict(), 'best_model.pth')
        
        st.success(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        return model