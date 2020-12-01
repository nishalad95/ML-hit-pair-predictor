import numpy as np
import pandas as pd
import os
import tarfile
from imblearn.under_sampling import RandomUnderSampler

class TrackData:

    
    def __init__(self, filename):
        self.filename = filename
        

    def read_and_merge_data(self):
        _, fileExtension = os.path.splitext(self.filename)
        if fileExtension != '.gz':
            raise TypeError("Data file provided of incorrect type, \n" +
                            "please provide data file of .tar.gz format")
        with tarfile.open(self.filename, "r:*") as tar:
            csv_files = [file for file in tar.getnames() if file.endswith('.csv')]
            df = pd.DataFrame()
            for file in csv_files:
                try:
                    # catch if file contents are empty
                    current_dataset = pd.read_csv(tar.extractfile(file), header=0, sep=',', low_memory=False)
                except:
                    continue
                df = pd.concat([df, current_dataset], ignore_index=True)
            
            tar.close()
            df = df.drop_duplicates()
            self.df = df
    
    
    def calculate_cott(self):
        self.df['r1'] = np.sqrt(self.df.x1**2 + self.df.y1**2)
        self.df['r2'] = np.sqrt(self.df.x2**2 + self.df.y2**2)
        self.df['r3'] = np.sqrt(self.df.x3**2 + self.df.y3**2)

        inner = (self.df.z2 - self.df.z1) / (self.df.r2 - self.df.r1)
        outer = (self.df.z3 - self.df.z2) / (self.df.r3 - self.df.r2)
        self.df['cot(t1)'] = inner
        self.df['cot(t2)'] = outer
        self.df['|cot(t1)|'] = np.abs( inner )    # inner doublet
        self.df['|cot(t2)|'] = np.abs( outer )    # outer doublet

        return self.df
    

    def generate_doublets(self, df, layers, label=False):
    
        # select the middle spacepoints in these layers of pixel region and succ track prop
        pixel_region_triplets = df[df.layer2.isin(layers)] 
    
        if label:
            pixel_region_triplets = pixel_region_triplets.loc[pixel_region_triplets.label == 1]

        # Good doublets:
        good_inner = pixel_region_triplets[pixel_region_triplets.inner_doublet == 1]
        good_outer = pixel_region_triplets[pixel_region_triplets.outer_doublet == 1]
        good_inner_doublets = good_inner[['weta2', '|cot(t1)|', 'z2', 'r2']]
        good_outer_doublets = good_outer[['weta2', '|cot(t2)|', 'z2', 'r2']]

        # Bad doublets:
        bad_inner = pixel_region_triplets[pixel_region_triplets.inner_doublet == 0]
        bad_outer = pixel_region_triplets[pixel_region_triplets.outer_doublet == 0]
        bad_inner_doublets = bad_inner[['weta2', '|cot(t1)|', 'z2', 'r2']]
        bad_outer_doublets = bad_outer[['weta2', '|cot(t2)|', 'z2', 'r2']]

        # Ground truth:
        good_inner_doublets.insert(0, 'target', 1)
        good_outer_doublets.insert(0, 'target', 1)
        bad_inner_doublets.insert(0, 'target', 0)
        bad_outer_doublets.insert(0, 'target', 0)
        
        # for tracking efficiency
        good_inner_doublets.insert(1, 'doublet', 'i')
        good_outer_doublets.insert(1, 'doublet', 'o')
        bad_inner_doublets.insert(1, 'doublet', 'i')
        bad_outer_doublets.insert(1, 'doublet', 'o')

        # Merge together:
        good_doublets = pd.concat([good_inner_doublets, good_outer_doublets], ignore_index=False, sort=True)
        good_doublets['|cot(t1)|'].update(good_doublets.pop('|cot(t2)|'))
        bad_doublets = pd.concat([bad_inner_doublets, bad_outer_doublets], ignore_index=False, sort=True)
        bad_doublets['|cot(t1)|'].update(bad_doublets.pop('|cot(t2)|'))
        doublets = pd.concat([good_doublets, bad_doublets], ignore_index=False)

        doublets.rename(columns={'|cot(t1)|':'tau', 'weta2': 'weta'}, inplace=True)
    
        return doublets
    

    def generate_endcap_doublets(self, df, pix_bar_layers, pix_ec_layers):
        # select triplets with inner spacepoint in barrel, middle spacepoint in endcap 
        # and successful track propagation
        pixel_region_triplets = df[(df.layer2.isin(pix_ec_layers)) & 
                                        (df.layer1.isin(pix_bar_layers)) &
                                        (df.label == 1)] 

        # Good doublets:
        good_inner = pixel_region_triplets[pixel_region_triplets.inner_doublet == 1]
        good_inner_doublets = good_inner[['weta1', '|cot(t1)|', 'z1', 'r1']]

        # Bad doublets:
        bad_inner = pixel_region_triplets[pixel_region_triplets.inner_doublet == 0]
        bad_inner_doublets = bad_inner[['weta1', '|cot(t1)|', 'z1', 'r1']]

        # Ground truth:
        good_inner_doublets.insert(0, 'target', 1)
        bad_inner_doublets.insert(0, 'target', 0)

        # For tracking efficiency
        good_inner_doublets.insert(1, 'doublet', 'i')
        bad_inner_doublets.insert(1, 'doublet', 'i')

        # # Merge together:
        doublets = pd.concat([good_inner_doublets, bad_inner_doublets], ignore_index=False)
        doublets.rename(columns={'|cot(t1)|':'tau', 'weta1':'weta'}, inplace=True)
        
        return doublets
 

    def weta_band(self, df, start, end, balanced=False):
        """ start < weta <= end""" 
        weta_band = df.loc[(df['weta'] > start) & (df['weta'] <= end)]
        
        if balanced:
            y = weta_band.loc[:,['target']].to_numpy().reshape(1, -1)[0]
            X = weta_band.drop(['target'], axis=1)
            # balance the data
            seed = int(np.random.uniform() * 100)
            rus = RandomUnderSampler(random_state=seed)
            X_balanced, y_balanced = rus.fit_resample(X, y)
            X_balanced['target'] = y_balanced
            
            weta_band = X_balanced
            
        return weta_band


    def downsample(self, df, max_size):
        seed = int(np.random.uniform() * 100)
        if len(df) > max_size:
            frac = max_size / (len(df))
            df = df.sample(frac=frac, replace=True, random_state=seed)

        return df
