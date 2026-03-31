#  Take raw images sitting in folders on disk → process them → 
#  serve them to the model in clean batches.

import os
from torch.utils.data import DataLoader, random_split 
from torchvision import datasets, transforms
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Transforms ─────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)), #resize
    transforms.RandomHorizontalFlip(), #horizonatl flip 
    transforms.RandomVerticalFlip(), #vertical flip 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), #vary brightness by 20%
    transforms.ToTensor(), #vary contrast by 20%
    transforms.Normalize(mean=config.MEAN, std=config.STD) #vary saturation by 20%
])

test_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),   #convert to tensor
     transforms.Normalize(mean=config.MEAN, std=config.STD) #no augmentation 
])

# ── Dataset Loader ─────────────────────────────────────────────
def get_dataloaders():
    # ImageFolder assigns labels based on subfolder names
    # FAKE/ → label 0    REAL/ → label 1
    full_train = datasets.ImageFolder(
        root = config.TRAIN_DIR, #path to train foloder
        transform = train_transform #applying the filters 
    )

    total = len(full_train)
    val_size = int(0.1*total)
    train_size = total - val_size

    train_dataset, val_dataset = random_split(full_train, 
                                              [train_size, val_size])
    
    #— only resize + normalize
    val_dataset.dataset.transform = test_transform

    #completely unseen data, used only at the very send 
    test_dataset = datasets.ImageFolder(
        root=config.TEST_DIR,        # path to test/ folder
        transform=test_transform     # no augmentation on test images
    )

    # DataLoader wraps dataset → serves images in batches during training
    train_loader = DataLoader(
        train_dataset, # our training dataset
        batch_size = config.BATCH_SIZE, # how many images per batch (32)
        shuffle = True, # shuffle every epoch so model doesn't memorize order
        num_workers = 2, # 2 parallel workers to load data faster
        pin_memory = True  # faster GPU transfer by keeping data in pinned memory
    )

    val_loader = DataLoader(
        val_dataset, # our validation dataset
        batch_size = config.BATCH_SIZE, # same batch size
        shuffle = False, # no shuffle — order doesn't matter for evaluation
        num_workers = 2, # parallel loading
        pin_memory = True # faster GPU transfer
    )
    
    test_loader = DataLoader(
        test_dataset,                # our test dataset
        batch_size=config.BATCH_SIZE,# same batch size
        shuffle=False,               # no shuffle for evaluation
        num_workers=2,               # parallel loading
        pin_memory=True              # faster GPU transfer
    )

    # summary for loading perfectly 
    print(f"Train size : {train_size}") # should be 90,000
    print(f"Val size   : {val_size}") # should be 10,000
    print(f"Test size  : {len(test_dataset)}") # should be 20,000
    print(f"Classes    : {full_train.classes}") # should be ['FAKE', 'REAL']

    return train_loader, val_loader, test_loader  # return all three loaders for use in training

