# Data-Analysis

This repository contains papers on various topics from the Data Analytics course. I find my work on clustering and building a bird image recommendation system the most interesting, which I will discuss in more detail below, but for more detailed information I recommend installing Jupyter Notebook, where I describe each step in more detail.

# Bird Image Clustering and Recommendation System

## Overview

This repository showcases a project that clusters a custom image dataset of **25 different bird species** and builds an interactive recommendation system. The goal is to group visually similar bird images and allow users to retrieve “similar” images based on a selected reference.

Key elements of the project:
1. **Feature Extraction** via a pre-trained VGG16 model.
2. **Dimensionality Reduction** using PCA, UMAP, and autoencoders.
3. **Clustering** with K-Means, HDBSCAN, and Agglomerative Clustering.
4. **Interactive GUI** to explore clusters and retrieve similar images.

This project is intended as a portfolio piece for a **Data Analyst** position. It demonstrates practical experience dealing with:
- Data preparation & cleaning
- Multiple clustering & dimensionality-reduction approaches
- Metrics for validation (Silhouette, ARI, AMI)
- Building a recommendation system that responds to user input
- User interface design with Tkinter + PIL

### Main Page
![MainPage](https://drive.google.com/uc?export=view&id=1XVP_w_X8PYVqurjfZt98ZxrvYYhk2f0z)

### Page with with the selected type of bird
![SelectedType](https://drive.google.com/uc?export=view&id=1cfxSFSGeki14T_UTBimdXAAgOjaOXj-9)

---

## 1. Project Background

### 1.1 Data Preparation & Feature Extraction

- **Dataset**: 
  - Self-collected or curated dataset of 25 bird species.
  - ~950 images per species.
  - Total: ~23,750 images.

- **VGG16 Feature Extraction**:
  - Images were passed through **VGG16** to obtain feature vectors.
  - This process is time-consuming, so the resulting features are provided in the `data/features/` folder to avoid the need to extract them again.

> **Note**: If you wish to replicate feature extraction, you will need a GPU or sufficient CPU resources and the correct versions of `TensorFlow` and `Keras`.

### 1.2 Environment and Library Versions

Due to various compatibility issues:
- **Python** 3.9 is used.
- **TensorFlow** 2.12.0 
- **Keras** 2.10 (installed explicitly if needed)
- **NumPy** 1.23
- plus standard data-science libraries like `pandas`, `scikit-learn`, `matplotlib`, `umap-learn`, and `hdbscan`.

---

## 2. Dimensionality Reduction Experiments

After feature extraction, each image had a high-dimensional feature vector. We tested multiple methods to reduce dimensionality:

1. **PCA**: Reduced features to 150 principal components.  
   - Found ~95% information loss, which severely impacted clustering quality.

2. **Low Variance & Correlation Thresholding**:  
   - Removed features with variance below 0.01 and those highly correlated (corr > 0.95).  
   - Preserved more useful information than PCA in this context.

3. **Autoencoders**:  
   - Tried different latent dimensions (200, etc.) and epochs (~20 epochs with MSE + ADAM).
   - Achieved a negative silhouette coefficient, indicating poor cluster separation.
   - Reconstruction error remained relatively high, ~80% data loss.

4. **t-SNE**:  
   - Chose perplexity=5, learning_rate=200, n_iter=3000.
   - Silhouette coefficient: ~0.34 for 5 clusters. Some improvement, but not fully satisfactory.

5. **UMAP**:  
   - Explored `n_neighbors` = [10, 30, 50, 70] and `min_dist` = [0.1, 0.5, 0.9].
   - Achieved best silhouette coefficients of ~0.4 for 5 clusters, significantly better than PCA or autoencoders.
   - Provided more visually distinct clusters in 2D space.

In the final pipeline, **UMAP** was chosen as the main dimensionality reduction for the clustering steps.

---

## 3. Clustering Methods and Results

We tested multiple clustering algorithms on the UMAP-reduced features:

1. **K-Means**:
   - Experimented with cluster counts ranging from 3 to ~25. 
   - Silhouette scores generally hovered between 0.25 and 0.4. 
   - Interestingly, cluster counts divisible by 4 showed better results in some runs; clusters 16, 20, 22, and 23 often performed well.

2. **HDBSCAN**:
   - Adaptive clustering that can label noisy points as outliers.
   - Provided moderate separation, but still around silhouette ~0.3–0.35.

3. **Agglomerative Clustering**:
   - Showed good alignment with true labels (visually and via metrics like ARI/AMI).
   - The final result was robust for different numbers of clusters.

### 3.1 Visualization & Metrics

- **Silhouette Score**: 
  - Gauges how close points in the same cluster are relative to points in other clusters.
- **ARI (Adjusted Rand Index)** & **AMI (Adjusted Mutual Information)**: 
  - Compare discovered clusters to the ground truth (bird species labels).
- **Homogeneity, Completeness, V-Measure**: 
  - Provide deeper insight into overlap vs. purity of clusters.

In general, methods reached a good score for all 25 species (which would effectively “classify” images exactly by species). However, the best runs often grouped images with similar poses, backgrounds, or body structures.

---

## 4. Building the Recommendation System

Once we could cluster images successfully in 2D or N-dimensional UMAP space, the next goal was to **retrieve “similar” images** for a user-selected example:

1. **Reduced Features**:  
   - We used the 2D or N-dimensional UMAP embeddings (depending on the chosen dimensionality).

2. **User Interface**:  
   - Implemented a GUI with **Tkinter** + **PIL** to display images and allow user interaction.
   - Users can select a “mode” (i.e., a certain combination of `n_neighbors`, `min_dist`, and number of clusters).
   - A random subset of images from each cluster is shown.

3. **Nearest Neighbor Search**:
   - When the user clicks on an image, the system:
     1. Identifies that image’s cluster (from the K-Means or other clustering labels).
     2. Builds a **BallTree** (or uses `NearestNeighbors`) on that cluster’s embeddings.
     3. Retrieves the top-K most similar images in that cluster by Euclidean distance (or a chosen metric).

4. **Exploration**:
   - The user can repeat the similarity search by clicking on another retrieved image, effectively browsing the dataset through visual similarity.

---


## 5. Key Observations & Lessons Learned

1. **High-Dimensional Data**  
   - Extracted VGG16 features can have thousands of dimensions; naive clustering is slow and often suboptimal without reduction.

2. **Dimensionality Reduction**  
   - PCA reduced dimensionality but lost crucial information.  
   - UMAP captured more nuanced structure, which led to higher silhouette and better visual clusters.

3. **Sensitivity to Hyperparameters**  
   - Both UMAP (n_neighbors, min_dist) and K-Means (n_clusters) significantly affect results.  
   - Iterative experimentation is key to fine-tune these parameters.

4. **Visualization & Validation**  
   - Silhouette, ARI, AMI metrics guide objective decisions.  
   - Visual inspection of clusters + reading images is essential for deeper insight.  

5. **Recommendation System**  
   - Combining cluster labels with a nearest-neighbor search inside clusters leads to an **interactive** approach.  
   - Letting users “click” and navigate the dataset fosters immediate feedback.

---

## 6. Future Improvements

- **Fine-Tuning with Other Pre-Trained Models** (e.g., EfficientNet, ResNet) to see if different feature extraction yields better clusters.
- **Hyperparameter Optimization** for autoencoders or other dimensionality reduction methods (e.g., custom autoencoder architectures).
- **Advanced Clustering**:
  - Try advanced methods like **Spectral Clustering** or **DBSCAN** with different distance metrics.
- **Scalability**:
  - Deploy the system with a back-end API + front-end, so others can test it online.

---

## 7. Acknowledgments

- Pre-trained VGG16 model courtesy of Keras.
- UMAP library: [https://umap-learn.readthedocs.io/](https://umap-learn.readthedocs.io/)
- HDBSCAN library: [https://hdbscan.readthedocs.io/](https://hdbscan.readthedocs.io/)


---

# Thanks for Visiting!


