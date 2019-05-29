def kmeans(points, n_clusters):
    # sample initial centroids
    sample = np.random.choice(
        len(points), n_clusters, replace=False
    )
    centroid = points[sample]
    
    loss = [-1, -2]
    while not np.allclose(*loss):
        # compute distance for each pair: point/centroid
        distance = [
            np.sqrt(((points - c) ** 2).sum(1)) for c in centroid
        ]
        # new loss
        loss = loss[1:] + [np.sum(distance)]
        # assign new clusters
        cluster = np.argmin(distance, axis=0)
        # update centroids by new cluster means
        for i in range(n_clusters):
            centroid[i] = np.mean(points[cluster == i], axis=0)
        
    return cluster
