import os
import pickle
from tqdm import tqdm

from utils import wsi_generator, slg_generator, commonality_g_constructor, data_transform

def main():
    organs = ['stomach', 'colon']
    modes = ['train', 'val', 'test']

    for organ in organs:
        for mode in modes:
            #loading tiles' features
            feature_path = f'files/features/{organ}_{mode}_features.pkl'
            with open(feature_path, 'rb') as f:
                feature_data = pickle.load(f)
            slide_names, conf_list, label_list, loc_list, slide_label = get_tiles_feature(feature_data)

            three_dim_graphs = []
            for idx, _ in enumerate(tqdm(slide_names)):
                # generating WSI-level graph
                wsi = wsi_generator(conf_list[idx], label_list[idx], loc_list[idx], slide_label[idx])

                # generating slice-level graphs
                slg = slg_generator(wsi)
                
                # constructing slice-level graphs to a commonality graph 
                if len(slg) > 1:
                    three_dim_g = commonality_g_constructor(slg, wsi.x)
                else: 
                    # if WSI contains one slice, 
                    # commonality graph = slice-level graph = WSI-level graph
                    three_dim_g = slg[0] 
                
                # tranforming commonality graph to training (data) form
                transform_g = data_transform(three_dim_g, wsi)
                three_dim_graphs.append(transform_g)
            
            # saving composed graph information
            save_dir = f'files/graphs/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            graph_save_path = os.path.join(save_dir, f'{organ}_{mode}_graphs.b')
            f = open(graph_save_path, 'wb')
            pickle.dump((slide_names, slide_label, three_dim_graphs), f)
            f.close()

def get_tiles_feature(data):
    slide_names = []
    conf_list = []
    label_list = []
    loc_list = []
    slide_label = []
    for slide_name in data.keys():
        # getting a list of slide names
        slide_names.append(slide_name)

        # getting a list of tiles' probability scores, labels, and locations
        tile_data_list = data[slide_name]
        conf = []
        for tile in tile_data_list:
            t_conf = tile['conf']
            conf.append(t_conf)
        conf_list.append(conf)

        label = []
        for tile in tile_data_list:
            t_Label = tile['label']
            label.append(t_Label)
        label_list.append(label)

        loc = []
        for tile in tile_data_list:
            t_loc = tile['location']
            loc.append(t_loc)
        loc_list.append(loc)

        # getting a list of slide labels
        if slide_name.startswith('D'):
            label = 0
        elif slide_name.startswith('M'):
            label = 1
        elif slide_name.startswith('N'):
            label = 2
        else:
            raise RuntimeError('Undefined slide type')
        slide_label.append(label)

    return slide_names, conf_list, label_list, loc_list, slide_label

if __name__ == '__main__':
    main()