# Content-Based-Image-Retrieval
- First step: calculate the histogram of query image
- 1. Color histograms
- - 1. Gray scale image with 8 bins
- - 2. Gray scale image with 256 bins
- - 3. RGB image with 256 bins
- 2. Local Binary Pattern (LBP) histograms
- - 1. Whole-image lbp histogram
- - 2. Grid-image with 32 bins for each 16x16 grid
- Second step: use distance function to find similar image and rank them
- - For example: euclidean, cityblock, correlation, canberra...
