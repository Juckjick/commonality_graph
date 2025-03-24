import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image

import Model.densenet as densenet

def main():
    # importing tiles data
    data_path = '' # path to tiles
    csv_path = data_path + 'data_distribution.csv' # path to csv (tiles' infomation)
    data = pd.read_csv(csv_path)
    
    organs = ['stomach', 'colon']
    modes = ['train', 'val', 'test']

    for organ in organs:
        organ_data = data[data['organ'] == organ]
        for mode in modes:
            # loading tiles' information
            mode_data = organ_data[organ_data['subset'] == mode]
            mode_data = mode_data[0:1000]
            transform_data = PatchDataset(mode_data, data_path)
            loader_data = torch.utils.data.DataLoader(transform_data, batch_size = 16, shuffle = False, num_workers = 0, pin_memory = False)

            save_dir = f'files/features/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # loading feature extractor
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            pkl_name = f'models/patch_classifier/{organ}'
            num_classes = 3
            model = densenet.densenet201(pretrained = True)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            model.load_state_dict(torch.load(pkl_name + '.pkl'))
            feat_extractor = model.to(device)

            # extracting and saving tiles' features
            extract_features(organ = organ, mode = mode, extractor = feat_extractor, loader = loader_data, device = device, save_path = save_dir, num_classes = num_classes)

def extract_features(organ, mode, extractor, loader, device, save_path, num_classes):
    extractor.eval()
    data = {}
    with torch.no_grad():
        for idx, batchdata in enumerate(tqdm(loader)):
            samples = batchdata[0].to(device) # transformed tile images
            tile_locations = batchdata[1] # tile locations
            slide_names = batchdata[2] # slide name (label_SlideName)
            tile_names = batchdata[3] # tile name

            conf, _ = extractor(samples)
            _, labels = torch.topk(conf, num_classes)
            conf_np = (F.softmax(conf, dim = 1).cpu().data.numpy()).tolist() # prediction probalities
            pred_labels = labels.cpu().data.numpy().tolist() # prediction (in label)

            for idx, slide_name in enumerate(slide_names):
                if slide_name not in data.keys():
                    data[slide_name] = []
                tile_conf = conf_np[idx]
                tile_label = pred_labels[idx]
                tile_loc = [int(tile_locations[0][idx]), int(tile_locations[1][idx])]
                tile_name = tile_names[idx]
                tile_data = {'conf': tile_conf, 'label': tile_label, 'location': tile_loc, 'tile_name': tile_name}
                data[slide_name].append(tile_data)
    
    if save_path != '':
        slide_save_path = os.path.join(save_path, f'{organ}_{mode}_features.pkl')
        with open(slide_save_path, 'wb') as f:
            pickle.dump(data, f)

class PatchDataset(Dataset):
    def __init__(self, data, data_path, img_resize = 256):
        self.data_path = data_path
        self.img_resize = img_resize
        self.patch_dirs = data['img_path'].tolist()
        self.slide_names = data['slide_name'].tolist()
        self.slide_labels = data['condition'].tolist()

    def __getitem__(self, index):
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean = [0.485, 0.456, 0.406], 
                        std = [0.229, 0.224, 0.225])
        ])
        # loading tile image and transforming it
        patch_dir = os.path.join(self.data_path, self.patch_dirs[index])
        with open(patch_dir, 'rb') as f:
            tile = Image.open(f).convert('RGB')
        tile = tile.resize((self.img_resize, self.img_resize))
        tile = transform(tile)
        
        # getting tile's location from its name (../sildename-100_62.jpg)
        location = self.patch_dirs[index].split("/")[-1].split("-")[-1].split(".")[0]
        tile_location = [location.split("_")[0], location.split("_")[1]] # [100, 62]

        slide_name = self.slide_labels[index] + '_' + self.slide_names[index]
        tile_name = self.patch_dirs[index]

        return tile, tile_location, slide_name, tile_name
    
    def __len__(self):
        return len(self.patch_dirs)

if __name__ == '__main__':
    main()