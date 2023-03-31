import os
import pandas as pd
from imutils import paths
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class CreateFile():
    
    def __init__(self, folders_dict, split_list, path_to_save, csv_names):
        
        self.folders_dict = folders_dict # Dictionary with folder_path:folder_label
        self.path_to_save = path_to_save # list with the full-paths to save the csv
        self.split_list = split_list # list with the sizes to split the dataframes. 0 -> test_size, 1 -> train_size
        self.csv_names = csv_names # list with the names of the csv files to save

    def images_to_csv(self, flag=True): # function that turns a folder of images to pd.Dataframe and then saves it as csv's file. 
                                        # if flag is False, it will not save the csv's files
        
        if flag:

            data_frames = [] # list that consists of all the dataframes

            # first turn images to pd.Dataframe:
            for folder_path, folder_label in self.folders_dict.items():
                df = self.write_image_paths_to_csv(folder_path=folder_path, folder_label=folder_label)
                data_frames.append(df)

            final_df = self._concat_df(data_frames) # then concatenate the dataframes in to a single one

            train_csv, val_csv, test_csv = self._train_test_split(final_df, \
                        test_size=self.split_list[0], train_size=self.split_list[1]) # then split in to training validation and test sets:

            # check if path exists:
            if not os.path.exists(self.path_to_save):
                os.makedirs(self.path_to_save)

            train_csv.to_csv(os.path.join(self.path_to_save, self.csv_names[0]))
            val_csv.to_csv(os.path.join(self.path_to_save, self.csv_names[1]))
            test_csv.to_csv(os.path.join(self.path_to_save, self.csv_names[2])) # then save the csv    
        
        else: 
            pass

    def _images_to_df(self, path, label_number): # function that turns a folder of images to pd.Dataframe

        image_paths = list(paths.list_images(path))  # getting all the image paths
        df = pd.DataFrame() 
        for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths), desc='Transforming folder to Dataframe.'):
            # 0 for real, 1 for fake
            label = label_number
            df.loc[i, 'image_path'] = image_path
            df.loc[i, 'label'] = label
        return df

    def _concat_df(self, dfs): # function that concatenates a list of dataframes
        return pd.concat(dfs, axis=0)

    def _train_test_split(self, df, test_size, train_size): # function that splits a dataframe in to training, validation and test sets
        train_df, test_df = train_test_split(df, test_size=test_size)
        train_df, val_df = train_test_split(train_df, train_size=train_size)
        return train_df, val_df, test_df
    

    def write_image_paths_to_csv(self, folder_path, folder_label):
        data = {'image_path': [], 'label': []}
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.jpg') or file_name.endswith('.png') or file_name.endswith('.jpeg'):
                data['image_path'].append(os.path.join(folder_path, file_name))
                data['label'].append(folder_label)
        df = pd.DataFrame(data)
        return df