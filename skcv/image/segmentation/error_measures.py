from cmath import sqrt
import numpy as np
import networkx as nx

from math import sqrt

def undersegmentation_error(partition, groundtruth,
                            tolerance=0.05):
    """ Computes the undersegmentation error defined as:
        ue(G_i) = (sum_{Area(S_i)} - area(G_i)) / area(G_i)
        where G_i is the groundtruth and
        S_i is the obtained partition
        The total error is the average accross regions

        Parameters
        ----------
        partition: (N,M) array
            array with obtained labels

        groundtruth: (N,M) array or list
            array(list with groundtruth labels

        tolerance: float, optional
            threshold to consider oversegmentation

        Returns
        -------
        The undersegmentation error
    """
    gt_list = [];
    if type(groundtruth) != list:
        gt_list.append(groundtruth)
    else:
        gt_list = gt_list + groundtruth

    # partition labels
    seg_labels = np.unique(partition)
    areas = {}

    for s_i in seg_labels:
        area = np.count_nonzero(partition == s_i)
        areas[s_i] = area

    # evaluate each groundtruth segmentation
    err = 0
    for segmentation in gt_list:
        gt_labels = np.unique(segmentation)
        err_s = 0
        # get error for each groundtruth region
        for g_i in gt_labels:

            # get groundtruth area
            area = np.count_nonzero(segmentation == g_i)

            # compute intersection
            total_area = 0.
            for s_i in seg_labels:
                n = np.count_nonzero((g_i == segmentation) *
                                     (partition == s_i))
                if n > tolerance*area:
                    total_area += areas[s_i]

            err_s += abs(total_area - area) / area

        err += err_s/len(gt_labels)

    return err / len(gt_list)

def segmentation_accuracy(partition, groundtruth):
    """ Computes the segmentation accuracy defined as:
        accu(G_i) = (sum_{Area(S_k) \in area(G_i)}) / area(G_i)
        where G_i is the groundtruth and
        S_k is the obtained partition where the majority of S_k is in G_i
        The total error is the average accross regions

        Parameters
        ----------
        partition: (N,M) array
            array with obtained labels

        groundtruth: (N,M) array or list
            array(list with groundtruth labels

        Returns
        -------
        The segmentation accuracy
    """
    gt_list = [];
    if type(groundtruth) != list:
        gt_list.append(groundtruth)
    else:
        gt_list = gt_list + groundtruth

    # partition labels
    seg_labels = np.unique(partition)

    # evaluate each groundtruth segmentation
    accu = 0
    for segmentation in gt_list:
        gt_labels = np.unique(segmentation)

        #find the area of each segment
        area = np.bincount(segmentation.astype(np.int).flatten())

        accu_s = 0

        # match each pixel to a groundtruth segment
        for s_k in seg_labels:
            coords = np.where(partition == s_k)

            #find the intersection
            intersection = np.bincount(segmentation[coords].flatten().astype(np.int))

            # get the maximum intersecting groundtruth segment
            g_i = np.argmax(intersection)
            accu_s += intersection[g_i] / area[g_i]

        accu += accu_s/len(gt_labels)

    return accu / len(gt_list)

def boundary_detection(partition, groundtruth, tolerance = 0.04):
    """ Measures boundary detection

        Parameters
        ----------
        partition: (N,M) array
            array with obtained labels

        groundtruth: (N,M) array or list
            array(list with groundtruth labels

        tolerance: float, optional
            maximum distance of considered boundaries relative
            to the diagonal

        Returns
        -------
        The precision recall boundaries
    """
    # dictionary holding contours and their status (matched/not matched)
    contours = {}
    gt_contours = {}

    # find horizontal contours for segmentation
    seg_hx, seg_hy = np.where(partition[:-1, :] != partition[1:, :])

    # find vertical contours for segmentation
    seg_vx, seg_vy = np.where(partition[:, :-1] != partition[:, 1:])

    # the third number reflects:
    # 0/1: horizontal/vertical contour
    # the forth number reflect
    # 0/1: segmentation/groundtruth contour
    for px,py in zip(seg_hx,seg_hy):
        contours[(px, py, 0, 0)] = 0

    for px,py in zip(seg_vx, seg_vy):
        contours[(px, py, 1, 0)] = 0

    # find horizontal contours for groundtruth
    seg_hx, seg_hy = np.where(groundtruth[:-1, :] != groundtruth[1:, :])

    # find vertical contours for groundtruth
    seg_vx, seg_vy = np.where(groundtruth[:, :-1] != groundtruth[:, 1:])

    # the third number reflects:
    # 0/1: horizontal/vertical contour
    # the forth number reflect
    # 0/1: segmentation/groundtruth contour
    for px,py in zip(seg_hx,seg_hy):
        gt_contours[(px, py, 0, 1)] = 0

    for px,py in zip(seg_vx, seg_vy):
        gt_contours[(px, py, 1, 1)] = 0

    # create a graph matching contours
    bipartite = nx.Graph()
    bipartite.add_nodes_from(contours)
    bipartite.add_nodes_from(gt_contours)

    diagonal = sqrt(partition.shape[0]**2 + partition.shape[1]**2)
    # maximum distance to search for
    D = int(tolerance * diagonal)
    for contour in contours:
        px = contour[0]
        py = contour[1]
        # find groundtruth contours around a neighborhood
        for x in range(px - D, px + D + 1):
            for y in range(py - D, py + D + 1):
                hcontour = (x, y, 0, 1)
                vcontour = (x, y, 1, 1)

                # add an edge if a contour is found
                if hcontour in bipartite:
                    bipartite.add_edge(contour, hcontour)
                if vcontour in bipartite:
                    bipartite.add_edge(contour, hcontour)

    # perform a matching
    # matches contains twice the matchings
    matches = nx.max_weight_matching(bipartite)

    print("Contours {0} and {1} matches {2}".format(len(contours),
          len(gt_contours), len(matches)))


    # find precision/recall values
    true_positives = len(matches)/2
    false_positives = len(contours) - len(matches)/2
    false_negatives = len(gt_contours) - len(matches)/2

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    return precision, recall

def explained_variation(img, partition):
    """ Computes the explained variation defined as:
    sum over voxels (\mu_i - \mu) / (\voxel - \mu)
    where \mu is the video mean and \mu_i is the region mean
    """
    # partition labels
    seg_labels = np.unique(partition)

    dimensions = img.shape

    #compute the color mean
    mu = np.zeros(dimensions[-1])

    #create an array to compute the mse error
    mse = np.zeros(dimensions[:-1])

    for i in range(dimensions[-1]):
        mu[i] = np.mean(img[..., i])
        mse += (img[..., i] - mu[i])**2

    #sum the error
    mse_error = np.sum(mse)

    #find the mse error for each
    mse_reg = 0
    for segment in seg_labels:
        coords = np.where(partition == segment)
        mu_i = np.mean(img[coords], axis=0)
        mse_reg += np.sum((img[coords] - mu_i)**2)

    return mse_reg / mse_error