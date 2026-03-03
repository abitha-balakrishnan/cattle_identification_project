# Dataset for Indian Cattle & Buffalo Breed Recognition

## Overview
This dataset contains images of various Indian cattle and buffalo breeds used for training the breed recognition model. The dataset is organized to support machine learning model training with proper train/validation/test splits.

## Dataset Structure
```
dataset/
├── train/
│   ├── Gir/
│   ├── Sahiwal/
│   ├── Ongole/
│   ├── Murrah/
│   ├── Jaffarabadi/
│   └── ...
├── val/
│   ├── Gir/
│   ├── Sahiwal/
│   ├── Ongole/
│   ├── Murrah/
│   ├── Jaffarabadi/
│   └── ...
└── test/
    ├── Gir/
    ├── Sahiwal/
    ├── Ongole/
    ├── Murrah/
    ├── Jaffarabadi/
    └── ...
```

## Target Breeds

### Indian Cattle Breeds
1. **Gir** - Origin: Gujarat, Known for high milk production, Distinctive curved horns
2. **Sahiwal** - Origin: Punjab region, Heat tolerant, Brownish red color
3. **Ongole** (Ongol) - Origin: Andhra Pradesh, White colored, Bullock strength
4. **Red Sindhi** - Origin: Sindh (Pakistan), Reddish brown, Good milch breed
5. **Tharparkar** - Origin: Rajasthan/Sindh, White/Grey, Dual purpose
6. **Kankrej** - Origin: Gujarat, Large size, White color
7. **Hariana** - Origin: Haryana, White/Black spotted, Draught purposes
8. **Rathi** - Origin: Rajasthan, Brown with white patches, Good milk yield

### Indian Buffalo Breeds
1. **Murrah** - Origin: Haryana, Black color, Highest milk yield
2. **Jaffarabadi** - Origin: Gujarat, Black/White spotted, Large size
3. **Surti** - Origin: Gujarat, Black/Brown, Good milk quality
4. **Mehsana** - Origin: Gujarat, Grey color, Good adaptability
5. **Nili-Ravi** - Origin: Punjab, Black/White, Dual purpose
6. **Pandharpuri** - Origin: Maharashtra, Black color, Good milk yield

## Data Collection Sources
- Government livestock portals
- Agricultural universities
- Veterinary research institutions
- Online repositories (with permission)
- Field surveys (ethically sourced)

## Image Specifications
- Format: JPEG, PNG
- Size: Minimum 224x224 pixels (will be resized during preprocessing)
- Quality: Clear, well-lit images preferred
- Angles: Various angles showing distinctive features

## Data Split Strategy
- Training: 70%
- Validation: 20% 
- Test: 10%

## Preprocessing Pipeline
- Resize to 224x224 pixels
- Normalize pixel values (0-1 range)
- Apply data augmentation (rotation, flip, brightness adjustment)
- Remove blurry or low-quality images
- Balance class distribution

## Quality Assurance
- Manual verification of breed labels
- Removal of duplicate images
- Verification of image clarity
- Ensuring representation of breed characteristics

## Dataset Report
- Total Images: [To be filled after collection]
- Classes: [To be filled after collection]
- Train Images: [To be filled after collection]
- Validation Images: [To be filled after collection]
- Test Images: [To be filled after collection]
- Average Image Size: [To be filled after collection]

## Ethical Considerations
- All images obtained ethically
- Proper permissions where required
- No copyrighted material used without permission
- Focus on publicly available or research-permitted images

## Data Augmentation Strategy
To increase dataset diversity and improve model generalization:
- Horizontal flip (random)
- Rotation (-15° to +15°)
- Brightness adjustment (±20%)
- Contrast adjustment (±20%)
- Zoom (0.9x to 1.1x)
- Shear transformation

## Expected Dataset Size
Target: Minimum 200-300 images per breed for reliable model training
Total: ~3,000-4,800 images across all breeds (initial phase)