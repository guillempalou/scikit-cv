import numpy as np

def undersegmentation_error(partition, groundtruth):
    """ Computes the undersegmentation error defined as:
        err(G_i) = (sum_{Area(S_i)} - area(G_i)) / area(G_i)
        where G_i is the groundtruth and
        S_i is the obtained partition
        The total error is the average accross regions

        Parameters
        ----------
        partition: (N,M) array
            array with obtained labels

        groundtruth: (N,M) array or list
            array(list with groundtruth labels

        Returns
        -------
        The undersegmentation error
    """
    gt_list = [];
    if type(groundtruth) != list:
        gt_list.append(groundtruth)
    else:
        gt_list = gt_list + groundtruth

    #partition labels
    seg_labels = np.unique(partition)


    #evaluate each groundtruth segmentation
    err = 0
    for segmentation in gt_list:
        gt_labels = np.unique(segmentation)
        err_s = 0
        #get error for each groundtruth region
        for g_i in gt_labels:

            #get groundtruth area
            area = np.count_nonzero(segmentation == g_i)

            #compute intersection
            total_area = 0
            for s_i in seg_labels:
                n = np.count_nonzero((segmentation == g_i) and
                                 (partition == s_i))
                total_area += n

            err_s += (total_area - area) / area

        err += err_s/len(gt_labels)

    return err / len(gt_list)

def segmentation_accuracy(partition, groundtruth):
    pass

def boundary_detection(partition, groundtruth):
    pass

def explained_variation(img, partition, groundtruth):
    pass