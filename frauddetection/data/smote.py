import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
import os
import matplotlib.pyplot as plt

def smote(T, N, k):
    n_minority_samples, numattrs = T.shape
    
    # If N is less than 100, randomize the minority class samples
    if N < 100:
        np.random.shuffle(T)
        n_minority_samples = (N // 100) * n_minority_samples  # Adjust the number of samples
        N = 100  # Normalize N to 100 for easier subsequent calculations
        
    N = N // 100  # Convert N to an integer factor
    Synthetic = np.zeros((N * n_minority_samples, numattrs))
    newindex = 0
    
    # Fit k-nearest neighbors
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(T)
    
    # Generate synthetic samples
    for i in range(n_minority_samples):
        # Compute k-nearest neighbors for the i-th minority sample
        nnarray = knn.kneighbors(T[i].reshape(1, -1), return_distance=False)[0]
        
        for _ in range(N):
            # Choose a random neighbor index from the k nearest neighbors
            nn = np.random.randint(1, k)  # Choose a random neighbor index (1 to k-1)
            
            for attr in range(numattrs):
                # Calculate the difference between the neighbor's and the current sample's attribute value
                dif = T[nnarray[nn]][attr] - T[i][attr]
                gap = np.random.rand()  # Generate a random gap value between 0 and 1
                # Create a synthetic sample by interpolating between the current sample and the neighbor
                Synthetic[newindex][attr] = T[i][attr] + gap * dif
            newindex += 1
    
    return Synthetic

def apply_smote(input):
    # Load data
    df = pd.read_csv(input, compression = 'gzip')
    time_amount = df[['Time', 'Amount']].values
    X = df.drop(['Time', 'Amount', 'Class'], axis = 1).values
    y = df['Class'].values
    
    # Scaling data
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(time_amount)
    scaled_features[:,1] = scaled_features[:, 1].astype(float)
    
    X_combined = np.hstack((scaled_features, X))
    
    minority_class_label = 1
    T = X_combined[y == minority_class_label]
    
    Synthetic = smote(T, N=200, k=5)
    
    X_resampled = np.vstack((X_combined, Synthetic))
    y_resampled = np.hstack((y, np.full(Synthetic.shape[0], minority_class_label)))
    
    return X_resampled, y_resampled
    
if __name__ == "__main__":
    input_csv = '~/creditfraud/creditfraud/data/raw/creditcard.csv.gz'
    X_resampled, y_resampled = apply_smote(input_csv)
    
    col_names = ['Time', 'Amount'] + [f'V{i}' for i in range(1, X_resampled.shape[1] -1)] + ['Class']
    
    resampled_data = pd.DataFrame(np.hstack((X_resampled, y_resampled.reshape(-1, 1))), columns=col_names)
    output_csv = '~/creditfraud/creditfraud/data/processed/resampled_creditcard.csv.gz'
    resampled_data.to_csv(output_csv, index = False,  compression='gzip')   
    
    

    
    
    

