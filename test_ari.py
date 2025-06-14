import h5py
import numpy as np
import scanpy as sc
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K
import pandas as pd
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import adjusted_rand_score

# 1. Load and Preprocess Data
def load_data(file_path):
    with h5py.File(file_path, 'r') as f:
        # Extract count matrix data
        data = f['matrix/data'][:]
        indices = f['matrix/indices'][:]
        indptr = f['matrix/indptr'][:]
        shape = f['matrix/shape'][:]
        
        # Create sparse matrix using CSR format
        counts = csr_matrix((data, indices, indptr), shape=(shape[1], shape[0]))
        gene_names = f['matrix/features/name'][:]
        
        # If the names are bytes, decode them
        if isinstance(gene_names[0], bytes):
            gene_names = [name.decode() for name in gene_names]
    
    return counts, gene_names


def preprocess_data(adata):
    # Filter out genes and cells
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=50)
    
    # Normalize data (total count normalization)
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # Log transform
    sc.pp.log1p(adata)
    return adata

# 2. Define Autoencoder Model with MINE at the Bottleneck Layer
def build_autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    
    # MINE: Mutual Information Neural Estimation
    bottleneck = Dense(encoding_dim, activation='relu', name='bottleneck')(encoded)
    
    # Autoencoder Decoder
    decoded = Dense(input_dim, activation='sigmoid')(bottleneck)
    
    # Model setup
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=bottleneck)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

# MINE Implementation
def mine_loss(x, y):
    # Mutual Information Estimation at the bottleneck layer
    mi = mutual_info_score(x.flatten(), y.flatten())
    return -K.log(K.abs(mi))

# 3. Apply Autoencoder with MINE
def apply_autoencoder(adata):
    input_dim = adata.X.shape[1]
    encoding_dim = 50  # Number of features in the latent space
    
    # Print dimensions before autoencoder
    print(f"Dimensions before autoencoder: {adata.X.shape}")

    # Build and train the autoencoder with MINE
    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
    
    # Train the autoencoder with MINE loss
    autoencoder.fit(adata.X.toarray(), adata.X.toarray(), epochs=50, batch_size=256, shuffle=True, verbose=2)
    
    # Get the latent features (encoded data)
    latent_features = encoder.predict(adata.X.toarray())
    adata.obsm['X_autoencoder'] = latent_features
    
    # Print dimensions after autoencoder
    print(f"Dimensions after autoencoder: {latent_features.shape}")
    
    return adata

def filter_latent_features_by_mi(adata, mi_threshold=0.01):
    latent_features = adata.obsm['X_autoencoder']
    cluster_labels = adata.obs['leiden'].astype(int)

    # Compute mutual information scores
    mi_scores = mutual_info_classif(latent_features, cluster_labels)

    # Filter latent features by threshold
    selected_indices = np.where(mi_scores > mi_threshold)[0]
    filtered_features = latent_features[:, selected_indices]

    print(f"Selected {filtered_features.shape[1]} latent features based on MI > {mi_threshold}")
    
    # Store filtered features
    adata.obsm['X_autoencoder_filtered'] = filtered_features
    return adata

# 4. Perform Clustering (Leiden) with Reduced Number of Clusters
def perform_clustering(adata, resolution=0.5, use_rep='X_autoencoder'):
    # Calculate neighbors based on the latent space
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=10)
    
    # Perform UMAP for visualization
    sc.tl.umap(adata)
    
    # Perform clustering using Leiden algorithm with resolution parameter
    sc.tl.leiden(adata, resolution=resolution)
    
    print(f"Leiden clustering complete with resolution = {resolution} using {use_rep}")
    return adata

def compute_ari_with_ground_truth(adata, ground_truth_csv):
    try:
        pred_df = adata.obs[['leiden']].copy()
        pred_df['barcode'] = adata.obs_names
        pred_df = pred_df.rename(columns={'leiden': 'cluster'})
        pred_df.to_csv("predicted_clusters.csv", index=False)

        # Load the ground truth cluster labels
        gt_df = pd.read_csv(ground_truth_csv)
        
        # Rename to match predicted DataFrame (barcode, cluster)
        gt_df = gt_df.rename(columns={'Barcode': 'barcode', 'Cluster': 'cluster'})

        # Merge predicted and ground truth by barcode
        merged_df = pd.merge(pred_df, gt_df, on='barcode', suffixes=('_pred', '_gt'))

        # Compute ARI
        ari_score = adjusted_rand_score(merged_df['cluster_gt'], merged_df['cluster_pred'])
        print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
        return ari_score

    except Exception as e:
        print(f"Error computing ARI: {e}")
        return None

# 5. Perform Differential Gene Expression (DGE) Analysis
def perform_dge_analysis(adata):
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")
    print("Differential Gene Expression (DGE) analysis complete.")
    return adata

# 6. Visualize Clusters
def visualize_results(adata, top_n=10):
    # Check if DGE results exist
    if "rank_genes_groups" not in adata.uns:
        print("Error: No rank_genes_groups found in adata. Did DGE analysis run successfully?")
        return None
    
    # Ensure that var_names are unique before proceeding
    adata.var_names_make_unique()

    # Extract top marker genes
    marker_genes = pd.DataFrame(adata.uns["rank_genes_groups"]["names"]).head(top_n)
    print("Top marker genes: ")
    print(marker_genes)
    
    # Plot UMAP with Leiden clusters
    sc.pl.umap(adata, color='leiden', save="_clusters.png")

    # Visualize ranked marker genes
    sc.pl.rank_genes_groups(adata, n_genes=top_n, save="_dge_marker_genes.png")
    
    # Generate dot plot
    sc.pl.dotplot(adata, var_names=marker_genes.iloc[:, 0].tolist(), groupby='leiden', save="_dotplot.png")
    print("Dot plot saved.")
    
    # Generate violin plots for marker genes
    top_genes = marker_genes.iloc[:, 0].tolist()  # Get top genes from the first column
    sc.pl.violin(adata, keys=top_genes[:4], groupby='leiden', save="_marker_genes_violin.png")
    print("Violin plot saved.")
    
    return marker_genes

    
# 7. Identify Disease-Related Genes

def identify_disease_related_genes(marker_genes_file, cancer_genes_file):
    """
    Finds common genes between marker genes and cancer genes.
    
    Parameters:
    - marker_genes_file (str): Path to the CSV file containing marker genes.
    - cancer_genes_file (str): Path to the CSV file containing universal cancer genes.
    
    Returns:
    - List of common genes.
    """
    try:
        # Load the CSV files with a different encoding (ISO-8859-1)
        marker_genes_df = pd.read_csv(marker_genes_file, encoding='ISO-8859-1')
        cancer_genes_df = pd.read_csv(cancer_genes_file, encoding='ISO-8859-1')
        
        # Assuming the gene names are in the first column of both files
        marker_genes = marker_genes_df.iloc[:, 0].tolist()
        cancer_genes = cancer_genes_df.iloc[:, 0].tolist()

        # Find common genes
        common_genes = list(set(marker_genes) & set(cancer_genes))
        
        # Print and return the common genes
        if common_genes:
            print("Disease-related genes found:")
            for gene in common_genes:
                print(gene)
        else:
            print("No Disease-related found.")
        
        return common_genes
    except Exception as e:
        print(f"Error processing files: {e}")
        return []

# 8. Plot Heatmap of Disease-Related Genes
def plot_marker_genes_heatmap(adata, marker_genes, disease_related_genes):
    # Filter marker genes to keep only disease-related ones
    disease_genes = [gene for gene in marker_genes.iloc[:, 0] if gene in disease_related_genes]
    
    # If no disease-related genes are found in the marker genes, print a message and return
    if not disease_genes:
        print("No disease-related genes found in the marker genes list.")
        return
    
    # Ensure that var_names are unique (important for gene names)
    adata.var_names_make_unique()
    
    # Create a matrix for the heatmap where rows are disease-related genes and columns are clusters
    sc.pl.heatmap(adata, var_names=disease_genes, groupby='leiden', cmap='coolwarm', 
                  use_raw=False, show=True, save="_heatmap_disease_related_genes.png")
    print("Heatmap of disease-related genes saved.")


# 9. Main Function to Execute the Workflow
def main_pipeline(h5_file_path,marker_genes_file,cancer_genes_file):
    # Load the dataset
    counts, gene_names = load_data(h5_file_path)
    adata = sc.AnnData(X=counts)

    # Add gene names as obs
    adata.var_names = gene_names

    # Preprocess the data
    adata = preprocess_data(adata)

    # Apply autoencoder
    adata = apply_autoencoder(adata)

    # TEMP: Cluster using unfiltered latent features to get leiden labels for MI filtering
    adata = perform_clustering(adata, resolution=0.5, use_rep='X_autoencoder') 

    # Filter latent features using mutual information with preliminary Leiden labels
    adata = filter_latent_features_by_mi(adata, mi_threshold=0.01)

    # RE-cluster using filtered latent features
    adata = perform_clustering(adata, resolution=0.5, use_rep='X_autoencoder_filtered')

    # Compute ARI using ground truth clusters
    ground_truth_csv = "D:\\4th yr project\\Visium_FFPE_Human_Cervical_Cancer_analysis\\analysis\\clustering\\graphclust\\clusters.csv"
    compute_ari_with_ground_truth(adata, ground_truth_csv)

    # DGE analysis and visualization
    adata = perform_dge_analysis(adata)

    # Visualize results
    marker_genes = visualize_results(adata)

    # Save results for further analysis
    marker_genes.to_csv("marker_genes.csv", index=False)
    print("Marker genes saved to 'marker_genes.csv'.")

    # Identify disease-related genes
    disease_related_genes = identify_disease_related_genes(marker_genes_file,cancer_genes_file)

    # Save disease-related genes
    with open("disease_related_genes.txt", "w") as f:
        for gene in disease_related_genes:
            f.write(f"{gene}\n")
    print("Disease-related genes saved to 'disease_related_genes.txt'.")

    # Plot the heatmap of marker genes and disease-related genes
    plot_marker_genes_heatmap(adata, marker_genes, disease_related_genes)

    return adata

# Execute the pipeline
if __name__ == "__main__":
    h5_file_path = "D:\\4th yr project\\Visium_FFPE_Human_Cervical_Cancer_raw_feature_bc_matrix.h5"
    marker_genes_file = "D:\\4th yr project\\marker_genes.csv"
    cancer_genes_file = "D:\\4th yr project\\disgenet_cancer_genes.csv"
    ground_truth_csv="D:\\4th yr project\\clusters.csv"
    main_pipeline(h5_file_path,marker_genes_file,cancer_genes_file)
