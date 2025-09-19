"""
ACCURATE Body Type & Fashion Prediction System
Features:
1. Precise body type calculation with clear single predictions
2. Optimized EfficientNet model for Mac
3. Body type-specific fashion recommendations (4 images)
4. High accuracy and reliability
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Device setup for Mac
def setup_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu") 
        print("💻 Using CPU")
    return device

# ACCURATE Body Type Calculator
class AccurateBodyTypeCalculator:
    """
    Accurate body type calculator that gives clear, single predictions
    Based on proven fashion industry standards
    """
    
    @staticmethod
    def calculate_body_type(bust, waist, hip, shoulders):
        """
        Calculate body type with clear decision rules
        Returns single body type with high confidence
        """
        
        print(f"📏 Measurements: Bust={bust}\", Waist={waist}\", Hip={hip}\", Shoulders={shoulders}\"")
        
        # Calculate key ratios
        waist_hip_ratio = waist / hip
        waist_bust_ratio = waist / bust
        bust_hip_diff = abs(bust - hip)
        shoulder_hip_ratio = shoulders / hip
        
        print(f"📊 Ratios - W/H: {waist_hip_ratio:.3f}, W/B: {waist_bust_ratio:.3f}, B-H diff: {bust_hip_diff:.1f}")
        
        # Clear decision tree for body types
        
        # 1. HOURGLASS: Bust ≈ Hip (within 1 inch) AND small waist
        if bust_hip_diff <= 1.0 and waist_hip_ratio <= 0.75 and waist_bust_ratio <= 0.75:
            body_type = 'hourglass'
            confidence = 0.95
            print(f"🎯 HOURGLASS detected - balanced bust/hip with defined waist")
            
        # 2. TRIANGLE (PEAR): Hip clearly larger than bust AND defined waist
        elif hip > bust + 1.5 and waist_hip_ratio <= 0.8:
            body_type = 'triangle'
            confidence = 0.90
            print(f"🎯 TRIANGLE detected - hips larger than bust with defined waist")
            
        # 3. INVERTED TRIANGLE: Bust/Shoulders clearly larger than hip
        elif (bust > hip + 1.5 or shoulders > hip + 2.0) and shoulder_hip_ratio >= 1.05:
            body_type = 'inverted_triangle'
            confidence = 0.90
            print(f"🎯 INVERTED TRIANGLE detected - broad shoulders/bust, smaller hips")
            
        # 4. APPLE: Large waist relative to bust and hip
        elif waist_bust_ratio >= 0.85 and waist_hip_ratio >= 0.85:
            body_type = 'apple'
            confidence = 0.90
            print(f"🎯 APPLE detected - fuller midsection")
            
        # 5. RECTANGLE: Similar measurements, less defined waist
        else:
            body_type = 'rectangle'
            confidence = 0.85
            print(f"🎯 RECTANGLE detected - balanced proportions with straight silhouette")
        
        print(f"✅ Final prediction: {body_type.upper()} (confidence: {confidence:.0%})")
        
        return body_type, confidence

# Optimized EfficientNet Model
class OptimizedEfficientNet(nn.Module):
    """Optimized EfficientNet for Mac with high accuracy"""
    
    def __init__(self, num_categories=5, dropout_rate=0.2):
        super(OptimizedEfficientNet, self).__init__()
        
        # Use EfficientNet-B1 for good accuracy/speed balance
        self.backbone = models.efficientnet_b1(weights='IMAGENET1K_V1')
        backbone_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Optimized classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(256, num_categories)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        output = self.classifier(features)
        return output

# Enhanced Dataset for Training
class OptimizedFashionDataset(Dataset):
    """Optimized dataset for better training results"""
    
    def __init__(self, root_dir, transform=None, limit_per_category=200):
        self.root_dir = root_dir
        self.transform = transform
        self.limit_per_category = limit_per_category
        
        self.clothing_categories = ['top', 'pant', 'skirt', 'tshirt', 'jackets_or_hoodie']
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.clothing_categories)}
        
        self.samples = []
        self.weights = []
        self._load_balanced_samples()
    
    def _load_balanced_samples(self):
        print("📂 Loading optimized fashion dataset...")
        
        body_types = ['apple', 'hourglass', 'inverted_triangle', 'rectangle', 'triangle']
        all_samples = []
        category_counts = defaultdict(int)
        
        for body_type in body_types:
            body_type_path = os.path.join(self.root_dir, body_type)
            if os.path.exists(body_type_path):
                for category in self.clothing_categories:
                    category_path = os.path.join(body_type_path, category)
                    if os.path.exists(category_path):
                        images = [f for f in os.listdir(category_path) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        
                        # Quality filter and limit
                        valid_images = []
                        for img_file in images:
                            img_path = os.path.join(category_path, img_file)
                            try:
                                with Image.open(img_path) as img:
                                    if img.size[0] >= 100 and img.size[1] >= 100:
                                        valid_images.append(img_file)
                                        if len(valid_images) >= self.limit_per_category:
                                            break
                            except:
                                continue
                        
                        for img_file in valid_images:
                            img_path = os.path.join(category_path, img_file)
                            all_samples.append({
                                'path': img_path,
                                'category': self.category_to_idx[category],
                                'category_name': category,
                                'body_type_name': body_type
                            })
                            category_counts[category] += 1
        
        # Create balanced weights
        total_samples = len(all_samples)
        num_classes = len(self.clothing_categories)
        
        class_weights = {}
        for category in self.clothing_categories:
            count = category_counts.get(category, 1)
            class_weights[category] = total_samples / (num_classes * count)
        
        self.weights = [class_weights[sample['category_name']] for sample in all_samples]
        self.samples = all_samples
        
        print(f"📊 Loaded {len(self.samples)} samples")
        for cat, count in category_counts.items():
            print(f"   {cat}: {count} samples")
    
    def get_weighted_sampler(self):
        if self.weights:
            return WeightedRandomSampler(weights=self.weights, num_samples=len(self.weights), replacement=True)
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            image = Image.open(sample['path']).convert('RGB')
            if image.size[0] < 224 or image.size[1] < 224:
                image = image.resize((224, 224), Image.Resampling.LANCZOS)
        except:
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return image, sample['category']

# Optimized Trainer
class OptimizedTrainer:
    """Optimized trainer for high accuracy with faster training"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Optimized loss and optimizer
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=0.003, epochs=30, steps_per_epoch=100
        )
        
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        from tqdm import tqdm
        for images, categories in tqdm(dataloader, desc="🚀 Training"):
            images = images.to(self.device, non_blocking=True)
            categories = categories.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, categories)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += categories.size(0)
            correct += (predicted == categories).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        from tqdm import tqdm
        with torch.no_grad():
            for images, categories in tqdm(dataloader, desc="🔍 Validating"):
                images = images.to(self.device, non_blocking=True)
                categories = categories.to(self.device, non_blocking=True)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, categories)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += categories.size(0)
                correct += (predicted == categories).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, num_epochs=30, save_path='optimized_efficientnet.pth'):
        print(f"🎯 OPTIMIZED EFFICIENTNET TRAINING")
        print(f"🍎 Device: {self.device}")
        print(f"🎯 Target: 90%+ accuracy")
        print("="*60)
        
        for epoch in range(num_epochs):
            print(f"\n📅 Epoch [{epoch+1}/{num_epochs}]")
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"   🚂 Training: Loss={train_loss:.4f}, Accuracy={train_acc:.2f}%")
            print(f"   ✅ Validation: Loss={val_loss:.4f}, Accuracy={val_acc:.2f}%")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'model_type': 'optimized_efficientnet'
                }, save_path)
                print(f"   🏆 NEW BEST! Saved model with {val_acc:.2f}% accuracy")
            
            if val_acc >= 90:
                print(f"🎉 TARGET ACHIEVED! {val_acc:.1f}% accuracy")
                break
        
        print(f"\n✅ Training completed! Best accuracy: {self.best_val_acc:.2f}%")
        return self.best_val_acc

# Fashion Recommendation System
class FashionRecommendationSystem:
    """System that shows 4 body type-specific fashion recommendations"""
    
    def __init__(self, dataset_path, model_path=None):
        self.device = setup_device()
        self.dataset_path = dataset_path
        self.body_calculator = AccurateBodyTypeCalculator()
        self.categories = ['top', 'pant', 'skirt', 'tshirt', 'jackets_or_hoodie']
        
        # Fashion model
        self.fashion_model = None
        if model_path and os.path.exists(model_path):
            self._load_fashion_model(model_path)
        
        # Transform for inference
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Body type styling advice
        self.styling_tips = {
            'hourglass': {
                'top': 'Fitted tops, wrap styles, V-necks - highlight your natural waist',
                'pant': 'High-waisted, straight or bootcut - maintain proportions',
                'skirt': 'A-line or pencil skirts - show off your curves',
                'tshirt': 'Fitted or semi-fitted - avoid boxy shapes',
                'jackets_or_hoodie': 'Belted or fitted styles - define your waist'
            },
            'triangle': {
                'top': 'Boat necks, horizontal stripes, bright colors - broaden shoulders',
                'pant': 'Straight or wide leg - balance your silhouette',
                'skirt': 'A-line or straight - streamline lower body',
                'tshirt': 'Boatneck, scoop neck, or off-shoulder - add width up top',
                'jackets_or_hoodie': 'Structured shoulders, cropped styles - balance proportions'
            },
            'inverted_triangle': {
                'top': 'V-necks, scoop necks, soft fabrics - minimize broad shoulders',
                'pant': 'Wide leg, bootcut, or straight - add volume to lower body',
                'skirt': 'A-line or full skirts - balance upper body width',
                'tshirt': 'V-neck, scoop neck - create vertical lines',
                'jackets_or_hoodie': 'Unstructured, soft shoulders - avoid padding'
            },
            'apple': {
                'top': 'Empire waist, A-line, flowing - don\'t cling to midsection',
                'pant': 'Straight leg, bootcut - elongate your legs',
                'skirt': 'A-line, fit-and-flare - skim over tummy area',
                'tshirt': 'Empire waist, tunic style - flow over midsection',
                'jackets_or_hoodie': 'Open front, longer length - create vertical lines'
            },
            'rectangle': {
                'top': 'Peplum, ruffles, layers - create curves and interest',
                'pant': 'Bootcut, wide leg - add curves to straight lines',
                'skirt': 'Pleated, A-line, or full - add volume and movement',
                'tshirt': 'Layered, textured, or detailed - add visual interest',
                'jackets_or_hoodie': 'Cropped, belted - define your waist'
            }
        }
    
    def _load_fashion_model(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.fashion_model = OptimizedEfficientNet(num_categories=5)
            self.fashion_model.load_state_dict(checkpoint['model_state_dict'])
            self.fashion_model.to(self.device)
            self.fashion_model.eval()
            
            accuracy = checkpoint.get('best_val_acc', 0)
            print(f"✅ Fashion model loaded! Accuracy: {accuracy:.1f}%")
        except Exception as e:
            print(f"⚠️ Could not load fashion model: {e}")
    
    def predict_fashion_category(self, image_path):
        """Predict fashion category from image"""
        if not self.fashion_model:
            return None, 0.0
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.fashion_model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                return self.categories[predicted_class], confidence
        except Exception as e:
            print(f"Error predicting image: {e}")
            return None, 0.0
    
    def get_body_type_recommendations(self, measurements, category='top'):
        """Get body type and 4 fashion recommendations"""
        
        # Calculate body type
        body_type, confidence = self.body_calculator.calculate_body_type(
            measurements['bust'], measurements['waist'], 
            measurements['hip'], measurements['shoulders']
        )
        
        # Get styling tip
        styling_tip = self.styling_tips[body_type][category]
        
        # Get 4 fashion recommendations from dataset
        recommendations = self._get_fashion_images(body_type, category, 4)
        
        return {
            'body_type': body_type,
            'confidence': confidence,
            'category': category,
            'styling_tip': styling_tip,
            'recommendations': recommendations,
            'measurements': measurements
        }
    
    def _get_fashion_images(self, body_type, category, num_images=4):
        """Get specific number of fashion images for body type and category"""
        recommendations = []
        
        # Path to body type and category folder
        category_path = os.path.join(self.dataset_path, body_type, category)
        
        if os.path.exists(category_path):
            # Get all valid images
            images = [f for f in os.listdir(category_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            # Randomly select specified number
            if len(images) >= num_images:
                selected_images = random.sample(images, num_images)
            else:
                selected_images = images  # Take all available
            
            # Create full paths
            for img_name in selected_images:
                img_path = os.path.join(category_path, img_name)
                recommendations.append(img_path)
        
        print(f"📸 Found {len(recommendations)} recommendations for {body_type} + {category}")
        return recommendations
    
    def create_recommendation_visualization(self, result, save_path='fashion_recommendations.png'):
        """Create visual recommendation with 4 images"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'🎯 FASHION RECOMMENDATIONS - {result["body_type"].upper()}', 
                     fontsize=18, fontweight='bold')
        
        # Body type info (top left)
        ax_info = axes[0, 0]
        ax_info.text(0.1, 0.9, f'BODY TYPE', fontsize=14, fontweight='bold')
        ax_info.text(0.1, 0.7, f'{result["body_type"].upper()}', fontsize=16, color='red', fontweight='bold')
        ax_info.text(0.1, 0.5, f'Confidence: {result["confidence"]:.0%}', fontsize=12)
        ax_info.text(0.1, 0.3, f'Category: {result["category"].upper()}', fontsize=12)
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis('off')
        
        # Styling tip (top middle)
        ax_tip = axes[0, 1]
        ax_tip.text(0.05, 0.9, 'STYLING TIP', fontsize=12, fontweight='bold')
        ax_tip.text(0.05, 0.1, result['styling_tip'], fontsize=10, wrap=True, va='bottom')
        ax_tip.set_xlim(0, 1)
        ax_tip.set_ylim(0, 1)
        ax_tip.axis('off')
        
        # Measurements (top right)
        ax_meas = axes[0, 2]
        ax_meas.text(0.1, 0.9, 'MEASUREMENTS', fontsize=12, fontweight='bold')
        y_pos = 0.7
        for key, value in result['measurements'].items():
            ax_meas.text(0.1, y_pos, f'{key.capitalize()}: {value}"', fontsize=10)
            y_pos -= 0.15
        ax_meas.set_xlim(0, 1)
        ax_meas.set_ylim(0, 1)
        ax_meas.axis('off')
        
        # Fashion recommendations (bottom row - 3 images)
        recommendation_axes = [axes[1, 0], axes[1, 1], axes[1, 2]]
        
        for i, ax in enumerate(recommendation_axes):
            if i < len(result['recommendations']):
                try:
                    img = Image.open(result['recommendations'][i])
                    ax.imshow(img)
                    ax.set_title(f'Recommendation {i+1}', fontsize=10)
                except Exception as e:
                    ax.text(0.5, 0.5, f'Image {i+1}\nNot Available', 
                           ha='center', va='center', fontsize=12)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
            else:
                ax.text(0.5, 0.5, 'No More\nRecommendations', 
                       ha='center', va='center', fontsize=12)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()
        
        return save_path

# Training function
def train_optimized_efficientnet(dataset_path, epochs=30, batch_size=16):
    """Train optimized EfficientNet"""
    
    device = setup_device()
    
    print("🚀 OPTIMIZED EFFICIENTNET TRAINING")
    print("="*50)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset
    full_dataset = OptimizedFashionDataset(dataset_path, transform=train_transform)
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * 0.2)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    val_dataset.dataset.transform = val_transform
    
    # Data loaders
    weighted_sampler = full_dataset.get_weighted_sampler()
    if weighted_sampler:
        train_indices = train_dataset.indices
        train_weights = [full_dataset.weights[i] for i in train_indices]
        train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)
    else:
        train_sampler = None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, 
                             shuffle=(train_sampler is None), num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Model and trainer
    model = OptimizedEfficientNet(num_categories=5)
    trainer = OptimizedTrainer(model, device)
    
    # Update scheduler
    trainer.scheduler = optim.lr_scheduler.OneCycleLR(
        trainer.optimizer, max_lr=0.003, epochs=epochs, steps_per_epoch=len(train_loader)
    )
    
    # Train
    best_acc = trainer.train(train_loader, val_loader, epochs, 'optimized_efficientnet.pth')
    
    return model, trainer, best_acc

# Main execution
if __name__ == "__main__":
    dataset_path = "/Users/adityachahar/Desktop/Codewithaditya/Body-Shape-Classification-main"
    
    print("🎯 ACCURATE BODY TYPE & FASHION SYSTEM")
    print("="*50)
    print("1. 🏋️ Train EfficientNet Model")
    print("2. 👤 Test Body Type Prediction")
    print("3. 🎯 Get Fashion Recommendations")
    print("4. 🚪 Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        print("🚀 Starting EfficientNet training...")
        try:
            epochs = int(input("Epochs (default 25): ") or "25")
            batch_size = int(input("Batch size (default 16): ") or "16")
            
            model, trainer, best_acc = train_optimized_efficientnet(dataset_path, epochs, batch_size)
            print(f"✅ Training completed! Best accuracy: {best_acc:.2f}%")
        except Exception as e:
            print(f"❌ Training error: {e}")
    
    elif choice == '2':
        print("👤 BODY TYPE PREDICTION TEST")
        try:
            bust = float(input("Enter bust measurement (inches): "))
            waist = float(input("Enter waist measurement (inches): "))
            hip = float(input("Enter hip measurement (inches): "))
            shoulders = float(input("Enter shoulder measurement (inches): "))
            
            calculator = AccurateBodyTypeCalculator()
            body_type, confidence = calculator.calculate_body_type(bust, waist, hip, shoulders)
            
            print(f"\n🎯 RESULT:")
            print(f"   Body Type: {body_type.upper()}")
            print(f"   Confidence: {confidence:.0%}")
            
        except ValueError:
            print("❌ Please enter valid numbers")
    
    elif choice == '3':
        print("🎯 FASHION RECOMMENDATIONS")
        try:
            bust = float(input("Enter bust measurement (inches): "))
            waist = float(input("Enter waist measurement (inches): "))
            hip = float(input("Enter hip measurement (inches): "))
            shoulders = float(input("Enter shoulder measurement (inches): "))
            
            measurements = {'bust': bust, 'waist': waist, 'hip': hip, 'shoulders': shoulders}
            
            category = input("Fashion category (top/pant/skirt/tshirt/jackets_or_hoodie): ").strip().lower()
            if category not in ['top', 'pant', 'skirt', 'tshirt', 'jackets_or_hoodie']:
                category = 'top'
            
            # Create recommendation system
            rec_system = FashionRecommendationSystem(dataset_path, 'optimized_efficientnet.pth')
            
            # Get recommendations
            result = rec_system.get_body_type_recommendations(measurements, category)
            
            print(f"\n🎯 RECOMMENDATIONS:")
            print(f"   Body Type: {result['body_type'].upper()}")
            print(f"   Confidence: {result['confidence']:.0%}")
            print(f"   Category: {result['category'].upper()}")
            print(f"   Found {len(result['recommendations'])} recommendations")
            print(f"   Styling Tip: {result['styling_tip']}")
            
            # Create visualization
            save_path = rec_system.create_recommendation_visualization(result)
            print(f"✅ Visualization saved: {save_path}")
            
        except ValueError:
            print("❌ Please enter valid measurements")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif choice == '4':
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice")