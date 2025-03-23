import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch
import faiss
import faiss.contrib.torch_utils
import numpy as np
from easydict import EasyDict
from easydict import EasyDict
from rank_bm25 import BM25Okapi
from tqdm import tqdm


def compute_rgcl_loss(batch,
                      labels,
                      feats,
                      train_dl,
                      model,
                      args,
                      reindex=False,
                      train_set=None,
                      sparse_retrieval_dictionary=None,
                      train_feats=None,
                      train_labels=None,
                      yes_token_id=None,
                      ):
    ids = batch["sample_ids"]
    batch_size = len(ids)

    labels = labels.bool()
    labels_inverse = ~labels
    # print(labels)
    # Matrix of size batch_size x batch_size
    label_matrix = torch.stack(
        [
            labels if labels[i] == True else labels_inverse
            for i in range(batch_size)
        ],
        axis=0,
    )
    # Bool to int conversion
    if args.no_pseudo_gold_positives == 0:
        label_matrix_positive = label_matrix.int()
    # print(label_matrix_positive)
    # FLip
    label_matrix_negative = (~label_matrix).int()
    # print(label_matrix_negative)
    # We then compute the number of in-batch positives and negatives per sample in the batch
    # vectors of sizes batch_size
    # Since the matrix is symmetric, use which dimension does not matter
    # -1 for minus the sample itself

    in_batch_positives_no = torch.sum(label_matrix, dim=1) - 1
    in_batch_negative_no = batch_size - in_batch_positives_no - 1

    # We then construct the similarity matrix by computing the
    # choice of loss function:
    # 1. cosine similarity
    # 2. Triplet loss
    # 3. Manhatten distance

    # We expand the feature matrix to a 3D tensor for vectorized computation
    # feats_expand Dimension: batch_size x feature_size x batch_size
    feats_expanded = feats.unsqueeze(
        2).expand(batch_size, -1, batch_size)

    if args.rgcl_metrics == "cos":
        cos = nn.CosineSimilarity(dim=1, eps=1e-8)

        # We compute the cosine similarity between each pair of features
        sim_matrix = cos(
            feats_expanded, feats_expanded.transpose(0, 2))
    elif args.rgcl_metrics == "ip":
        sim_matrix = torch.sum(
            feats_expanded * feats_expanded.transpose(0, 2), dim=1) / args.proj_dim
    elif args.rgcl_metrics == "l2":
        # l2 = nn.PairwiseDistance(p=2, eps=1e-8)
        # Poor vectorized implementation for pairwise distance
        # We use mse instead for vectorized computation

        """
        l2 = torch.nn.MSELoss(reduction='none')
        sim_matrix = l2(
            feats_expanded, feats_expanded.transpose(0, 2)).sum(dim=1) / args.proj_dim
        """
        # sim_matrix = torch.sum(torch.square((feats_expanded - feats_expanded.transpose(0, 2))), dim=1)
        sim_matrix = compute_l2(feats_expanded, feats_expanded.transpose(
            0, 2), normalise=args.norm_feats_loss, sum_dim=1, sqrt=args.l2_sqrt)
        # Add a negative sign here to account for the fact that
        # L2 is a distance measure, not a similarity measure, larger is more distant (dissimilar),
        # Where in similarity measure, larger is more similar

        # SQRT here gives NAN, thus we minimize the square of the L2 distance
        sim_matrix = - sim_matrix / args.proj_dim

    # The diagonal of the similarity matrix is 1,
    # which is the similarity of the same pair
    # Thus replace it with 0
    sim_matrix.fill_diagonal_(0)

    # We compute the loss matrix by multiplying the similarity matrix
    in_batch_negative_loss = sim_matrix * label_matrix_negative

    if args.no_pseudo_gold_positives == 0:
        in_batch_positives_loss = sim_matrix * label_matrix_positive
    else:
        # If we use pseudo gold positives, we do not use in batch positives
        # We set it to a matrix of zeros to make sure the contrastive loss can still use the same code
        in_batch_positives_loss = torch.zeros(
            batch_size, batch_size).to(args.device).to(feats.dtype)
    # V4 implementation, doing the loss with the mask rather than detect if there is nan and set to zero:
    # Pick out the non-zero terms (gives 1), mask out the zero terms (gives 0)
    neg_mask = in_batch_negative_loss != 0
    # Make sure neg_mask has same dtype as in_batch_negatives_loss
    neg_mask = neg_mask.to(feats.dtype)
    # Dim batch_size, count the number of zeros for each sample in the batch,
    neg_zero_count = (neg_mask == 0).sum(dim=1)

    # However, if all the terms are zero, we will get nan due to zero division,
    # We will form a further mask to only operate on the sample with at least one non-zero term
    neg_zero_count_zero_mask = torch.zeros(
        batch_size, device=args.device, dtype=feats.dtype) != in_batch_negative_no

    in_batch_negative_loss_sum = torch.zeros(
        batch_size, device=args.device, dtype=feats.dtype)
    in_batch_negative_loss_sum[neg_zero_count_zero_mask] = torch.sum(
        in_batch_negative_loss[neg_zero_count_zero_mask], dim=1) / neg_mask.sum(dim=1)[neg_zero_count_zero_mask]

    # Only use in-batch positive if we do not use pseudo gold positive samples
    if args.no_pseudo_gold_positives == 0:
        # V4 implementation, doing the loss with the mask rather than detect if there is nan and set to zero:
        # Pick out the non-zero terms (gives 1), mask out the zero terms (gives 0)
        pos_mask = in_batch_positives_loss != 0
        pos_mask = pos_mask.to(feats.dtype)
        # Dim batch_size, count the number of zeros for each sample in the batch,
        pos_zero_count = (pos_mask == 0).sum(dim=1)
        # However, if all the terms are zero, we will get nan due to zero division,
        # We will form a further mask to only operate on the sample with at least one non-zero term
        pos_zero_count_zero_mask = pos_zero_count != in_batch_positives_no

        in_batch_positives_loss_sum = torch.zeros(
            batch_size, device=args.device, dtype=feats.dtype)
        in_batch_positives_loss_sum[pos_zero_count_zero_mask] = torch.sum(
            in_batch_positives_loss[pos_zero_count_zero_mask], dim=1) / pos_mask.sum(dim=1)[pos_zero_count_zero_mask]

    # If we use pseudo gold positives, we do not use in batch positives
    else:
        in_batch_positives_loss_sum = 0
    if args.in_batch_loss:
        in_batch_loss = in_batch_negative_loss_sum - in_batch_positives_loss_sum
    else: 
        in_batch_loss = torch.Tensor([0.0]).to(args.device).to(feats.dtype)
    # Sanity check

    # ----------------- Hard Negative Retrieval and Pseudo Gold Positive -----------------
    # retrieve hard negatives and pseudo gold with Dense retrieval

    # Only hard negative
    # print("start to retrieve hard negatives and pseudo gold")
    if args.sparse_dictionary is None:
        if args.hard_negatives_loss and args.no_pseudo_gold_positives == 0:
            (
                hard_negative_features,
                hard_negative_scores,
                train_feats,
                train_labels,
            ) = dense_retrieve_hard_negatives_pseudo_positive(
                train_dl,
                feats,
                labels,
                model,
                largest_retrieval=args.no_hard_negatives,
                args=args,
                train_feats=train_feats,
                train_labels=train_labels,
                reindex=reindex,
                yes_token_id=yes_token_id
            )
            hard_negative_features = hard_negative_features.to(
                args.device).to(feats.dtype)
        # Both hard negative and pseudo gold,
        # In default we will consider hard negative, which is key
        # to the good performance.
        # But if we want to test without hard negative, this is also fine
        # We can just ignore the hard negative features and scores
        elif args.no_pseudo_gold_positives > 0:
            (
                hard_negative_features,
                hard_negative_scores,
                pseudo_positive_features,
                pseudo_positive_scores,
                train_feats,
                train_labels,
            ) = dense_retrieve_hard_negatives_pseudo_positive(
                train_dl,
                feats,
                labels,
                model,
                largest_retrieval=args.no_pseudo_gold_positives,
                args=args,
                train_feats=train_feats,
                train_labels=train_labels,
                reindex=reindex,
                yes_token_id=yes_token_id
            )
            hard_negative_features = hard_negative_features.to(
                args.device).to(feats.dtype)
            pseudo_positive_features = pseudo_positive_features.to(
                args.device).to(feats.dtype)
        else:
            pass
    # For sparse retrieval,
    # we always retrieve both hard negatives and pseudo gold
    # Since no computation will be saved
    # by only retrieving hard negatives/pseudo gold
    else:
        (hard_negative_features,
            pseudo_positive_features,
         ) = sparse_retrieve_hard_negatives_pseudo_positive(
            ids,
            labels,
            args.train_set,
            model,
            args.sparse_retrieval_dictionary,
            args,
        )
        hard_negative_features = hard_negative_features.to(
            args.device).to(feats.dtype)
        pseudo_positive_features = pseudo_positive_features.to(
            args.device).to(feats.dtype)

    # for hard negative loss
    if args.hard_negatives_loss:
        # Now we have the hard negatives features, we compute the loss

        # hard_negative_scores size batch_size, largest_retrieval

        # We compute the similarity matrix between the hard negatives and the original features
        # The dimension of hard_negative_features is batch_size x no_hard_negatives x dim
        # The dimension of original feats is batch_size x dim
        # We thus need to expand the original feats to batch_size x no_hard_negatives x embed_dim/hidden_dim
        feats_expanded = feats.unsqueeze(1).expand(
            batch_size, args.no_hard_negatives, -1
        )

        # The returned hard_negative_features might contain all zero embeddings for some samples,
        # We need to discard them in the loss computation
        # What we need to do is to construct a mask to zero out the loss for those samples

        # For simplicity, we only check if the first dimension is zero in the feature embedding
        # The mask is batch_size x no_hard_negatives, 1 if embedding non zero, 0 if embedding zero,
        # Thus we can multiply the mask with the loss.
        #zeroLoss_mask = hard_negative_features[:, :, 0] != 0

        # 2024.12.07 update, the above method is not correct, since the first dimension can be zero for some samples
        # Instead, we will sum the sum of the value of the embeddings
        # If the sum is zero, then we will set the mask to zero
        zeroLoss_mask = torch.sum(hard_negative_features, dim=2) != 0

        if args.rgcl_metrics == "cos":
            # Compute loss
            # Loss is batch_size x no_hard_negatives
            # print(hard_negative_scores)
            # we compute the cosine similarity
            cos_hard = nn.CosineSimilarity(dim=2, eps=1e-8)
            hard_loss = zeroLoss_mask * cos_hard(
                feats_expanded, hard_negative_features)
            # print(hard_loss.shape)
            # print(hard_loss)
        elif args.rgcl_metrics == "ip":
            # Compute loss
            # Loss is batch_size x no_hard_negatives
            hard_loss = zeroLoss_mask * torch.sum(
                feats_expanded * hard_negative_features, dim=2
            ) / args.proj_dim

        elif args.rgcl_metrics == "l2":

            """
            l2_hard = torch.nn.MSELoss(reduction='none')
            hard_loss = l2_hard(feats_expanded,
                                hard_negative_features).sum(dim=2)
            """
            # hard_loss = zeroLoss_mask * torch.sum(torch.square((feats_expanded - hard_negative_features)), dim=2)
            hard_loss = compute_l2(feats_expanded, hard_negative_features,
                                   normalise=args.norm_feats_loss, sum_dim=2, sqrt=args.l2_sqrt)
            hard_loss *= zeroLoss_mask
            """print("feats_expanded:", feats_expanded)
            print("hard negative features:", hard_negative_features)
            print("hard negative features shape:", hard_negative_features.shape)
            print("hard loss:", hard_loss)"""
            # SQRT gives NAN, thus we minimize the square of the L2 distance
            hard_loss = - hard_loss / args.proj_dim

        # For contrastive loss, we take mean during the loss computation
        if args.rgcl_loss != "contrastive":
            # Hard loss batch_size * no_hard_neg -> batch_size
            hard_loss = torch.sum(hard_loss, dim=1)

    # If not using hard negative, set to 0
    else:
        # hard_loss = 0
        hard_loss = torch.tensor([0.0], device=args.device)

    # for pseudo gold loss
    if args.no_pseudo_gold_positives != 0:
        # Now we have the pseudo gold positive features, we compute the loss
        # pseudo_positive_scores size: batch_size, args.no_pseudo_gold_positives

        feats_expanded = feats.unsqueeze(1).expand(
            batch_size, args.no_pseudo_gold_positives, -1
        )
        if args.rgcl_metrics == "cos":
            # Compute loss
            # Loss is batch_size x no_pseudo_gold_positives
            # print(pseudo_positive_scores)
            # we compute the cosine similarity
            cos_pseudo_gold = nn.CosineSimilarity(dim=2, eps=1e-8)
            pseudo_gold_loss = cos_pseudo_gold(
                feats_expanded, pseudo_positive_features)
            # print(pseudo_gold_loss.shape)
            # print(pseudo_gold_loss)
        elif args.rgcl_metrics == "ip":
            # Compute loss
            # Loss is batch_size x no_hard_negatives
            pseudo_gold_loss = torch.sum(
                feats_expanded * pseudo_positive_features, dim=2
            ) / args.proj_dim

        elif args.rgcl_metrics == "l2":

            # pseudo_gold_loss = torch.sum(torch.square((feats_expanded - pseudo_positive_features)), dim=2)
            pseudo_gold_loss = compute_l2(
                feats_expanded, pseudo_positive_features, normalise=args.norm_feats_loss, sum_dim=2, sqrt=args.l2_sqrt)

            # SQRT gives NAN, thus we minimize the square of the L2 distance
            pseudo_gold_loss = - pseudo_gold_loss / args.proj_dim

        # For contrastive loss, we take mean during the loss computation
        if args.rgcl_loss != "contrastive":

            pseudo_gold_loss = torch.mean(pseudo_gold_loss, dim=1)

    # if not using psedo gold, set to 0
    else:
        pseudo_gold_loss = torch.tensor([0.0], device=args.device)

    if args.rgcl_loss == "naive":
        # Take mean on batch-sample level
        total_loss = torch.mean(in_batch_loss + hard_loss - pseudo_gold_loss)
    elif args.rgcl_loss == "triplet":
        total_loss = torch.mean(torch.relu(
            in_batch_loss + hard_loss - pseudo_gold_loss + args.triplet_margin))

    elif args.rgcl_loss == "contrastive":

        neg_mask = in_batch_negative_loss != 0

        neg_zero_count = (neg_mask == 0).sum(dim=1)

        neg_zero_count_zero_mask = torch.zeros(
            batch_size, device=args.device, dtype=feats.dtype) != in_batch_negative_no
        in_batch_negative_loss_tmp = torch.zeros(
            batch_size, device=args.device, dtype=feats.dtype)
        in_batch_negative_loss_tmp[neg_zero_count_zero_mask] = (torch.exp(in_batch_negative_loss[neg_zero_count_zero_mask]).sum(
            dim=1) - neg_zero_count[neg_zero_count_zero_mask])
        in_batch_negative_loss = in_batch_negative_loss_tmp
        """print(in_batch_negative_no)
        print(neg_zero_count)
        print(neg_zero_count_zero_mask)
        print(neg_mask.sum(dim=1))
        print((neg_mask.sum(dim=1))[neg_zero_count_zero_mask])"""

        if args.no_hard_negatives != 0:
            # Dim batch size x no_hard_negatives
            hard_neg_mask = hard_loss != 0

            # Dim batch_size
            hard_zero_count = (hard_neg_mask == 0).sum(dim=1)
            # Constract this matrix to avoid zero division error
            # hard_zero_count_zero_mask = hard_zero_count == 0
            hard_zero_count_zero_mask = hard_zero_count != args.no_hard_negatives
            # initialise all zero matrix for hard loss
            hard_loss_tmp = torch.zeros(
                batch_size, device=args.device, dtype=feats.dtype)
            # We need to count the number of zero terms to discard them in the loss computation,
            # Since zero terms gives exp(0) = 1, we will delete the the number of zeros to discard the zero term
            hard_loss_tmp[hard_zero_count_zero_mask] = (torch.exp(hard_loss[hard_zero_count_zero_mask]).sum(
                dim=1) - hard_zero_count[hard_zero_count_zero_mask]) / (hard_neg_mask.sum(dim=1))[hard_zero_count_zero_mask]
            hard_loss = hard_loss_tmp

        """print(hard_zero_count)
        print(hard_zero_count_zero_mask)
        print((hard_neg_mask.sum(dim=1)))
        print((hard_neg_mask.sum(dim=1))[hard_zero_count_zero_mask])"""

        # If we dont have pseudo gold positives, we use the in batch positives
        if args.no_pseudo_gold_positives == 0:

            in_batch_positives_loss = torch.mean(
                torch.exp(in_batch_positives_loss), dim=1)
            loss = - torch.log(in_batch_positives_loss /
                               (in_batch_negative_loss + in_batch_positives_loss + hard_loss))
        # If we have pseudo gold positives, we use the pseudo gold positives rather than the in batch positives
        else:

            pseudo_gold_loss = torch.mean(torch.exp(pseudo_gold_loss), dim=1)

            loss = - torch.log(pseudo_gold_loss / (hard_loss +
                               pseudo_gold_loss + in_batch_negative_loss))

        total_loss = torch.mean(loss)

    return total_loss, torch.mean(in_batch_loss), torch.mean(hard_loss), torch.mean(pseudo_gold_loss), train_feats, train_labels


def compute_l2(feats_1, feats_2, normalise=False, sum_dim=1, sqrt=False, eps=1e-5):
    """Compute L2 loss."""
    l2_loss = 0
    if normalise:
        feats_1 = torch.nn.functional.normalize(feats_1, dim=sum_dim)
        feats_2 = torch.nn.functional.normalize(feats_2, dim=sum_dim)
    if not sqrt:
        l2_loss = torch.sum(torch.square((feats_1 - feats_2)), dim=sum_dim)
    else:
        l2_loss = torch.sqrt(torch.sum(torch.square(
            (feats_1 - feats_2)), dim=sum_dim) + torch.finfo(torch.float32).tiny)

    return l2_loss


# This function is implemented for the baseline experiment of DPR dpr_baseline.py
# Given the dense vectors of the image and text,
# we retrieve the top k image and text MM pairs
def retrieve_topk(
    database, query, alpha=1.0, beta=1.0, largest_retrieval=100, threshold=0.2
):
    """
    This function retrieve the top k image and text MM pairs for val/test
    with the given dense vectors

    input:
    database: the database set
    query: the query set
    alpha: the weight for image retrieval
    beta: the weight for text retrieval
    largest_retrieval: the largest number of retrieval neighbours
    threshold: the threshold for the similarity score

    The retrieved neighbours has to be larger than the threshold, if more than
    #largest_retrieval neighbours are larger than the threshold, then only the first
    #largest_retrieval neighbours are returned

    """

    ids,  img_feats, text_feats, labels = database
    q_ids, q_img_feats, q_text_feats, q_labels = query

    # Normalize the features
    img_feats_norm = img_feats.cpu().numpy().astype("float32")
    faiss.normalize_L2(img_feats_norm)
    text_feats_norm = text_feats.cpu().numpy().astype("float32")
    faiss.normalize_L2(text_feats_norm)

    q_img_feats_norm = q_img_feats.cpu().numpy().astype("float32")
    faiss.normalize_L2(q_img_feats_norm)
    q_text_feats_norm = q_text_feats.cpu().numpy().astype("float32")
    faiss.normalize_L2(q_text_feats_norm)
    feats = np.concatenate(
        (alpha * img_feats_norm, beta * text_feats_norm), axis=1)
    q_feats = np.concatenate(
        (alpha * q_img_feats_norm, beta * q_text_feats_norm), axis=1
    )

    # Get the dimension of the features
    dim = feats.shape[1]
    # Initialize the index
    index = faiss.IndexFlatL2(dim)
    index.add(feats)
    D, I = index.search(q_feats, largest_retrieval)

    logging_dict = EasyDict()

    for i, row in enumerate(D):
        # a list to record the ids of the retrieved example
        retrieved_ids = []
        # a list to record the similarity scores of the retrieved example
        retrieved_scores = []
        # a list to record the retrieved example's label
        retrieved_label = []
        for j, value in enumerate(row):
            # You have to retrieve at least one, no matter what the similarity score is
            if j == 0:
                retrieved_ids.append(ids[I[i, j]])
                retrieved_scores.append(value)
                retrieved_label.append(labels[I[i, j]].item())
            # if image is similar
            else:
                if value < threshold:
                    # for the temp list, we use the image ids rather than the ordered number
                    retrieved_ids.append(ids[I[i, j]])
                    retrieved_scores.append(value)
                    retrieved_label.append(labels[I[i, j]].item())
                # if larger than threshold,
                # then we can break the inside loop,
                # since the rest of the values are larger than the threshold
                else:
                    break
        # Record the number of images retrieved for each query
        no_retrieved = len(retrieved_ids)

        logging_dict[q_ids[i]] = {
            "no_retrieved": no_retrieved,
            "retrieved_ids": retrieved_ids,
            "retrieved_scores": retrieved_scores,
            "retrieved_label": retrieved_label,
        }

    return logging_dict


# Structure the sparse retrieval data
def get_sparse_data_FB(
    img_dict,
    gt_train,
    gt_dev,
    gt_test_seen,
    gt_test_unseen,
    attribute=True,
    objects_conf_threshold=None,
):
    retrieve_train = {}
    retrieve_val = {}
    retrieve_test_seen = {}
    retrieve_test_unseen = {}
    gt_train.set_index("id", inplace=True)
    gt_dev.set_index("id", inplace=True)
    gt_test_seen.set_index("id", inplace=True)
    gt_test_unseen.set_index("id", inplace=True)

    # iterate through the image dictionary
    for img_id in img_dict:
        # get the image id
        # img_id = img_dict[img_id]["img_id"]

        # get the object names
        object_names = img_dict[img_id]["object_names"]
        # get the object confidences
        objects_conf = img_dict[img_id]["objects_conf"]

        # get the attribute names
        if attribute:
            attribute_names = img_dict[img_id]["attribute_names"]
        else:
            # if attribute is false, use empty list
            attribute_names = [""] * len(object_names)
        # get the attribute confidences
        # attrs_conf = img_dict[img_id]["attrs_conf"]

        if objects_conf_threshold:
            # Since the confidences are sorted, we can just take the first n
            num_objects = np.sum(objects_conf >= objects_conf_threshold)
            # If all the confidences are smaller than the threshold,
            # then we just use the first one
            if num_objects == 0:
                num_objects = 1
            object_names = object_names[:num_objects]
            attribute_names = attribute_names[:num_objects]

        # Concat the object and attribute names for each object
        attobject_list = [
            obj + " " + attr for obj, attr in zip(object_names, attribute_names)
        ]

        # get the ground truth captions and concat with the object and attribute names
        if img_id in gt_train.index:
            #
            retrieve_train[img_id] = {
                "text": gt_train.loc[img_id]["text"] + " " + " ".join(attobject_list),
                "label": gt_train.loc[img_id]["label"],
            }
        elif img_id in gt_dev.index:
            retrieve_val[img_id] = {
                "text": gt_dev.loc[img_id]["text"] + " " + " ".join(attobject_list),
                "label": gt_dev.loc[img_id]["label"],
            }
        elif img_id in gt_test_seen.index:
            retrieve_test_seen[img_id] = {
                "text": gt_test_seen.loc[img_id]["text"]
                + " "
                + " ".join(attobject_list),
                "label": gt_test_seen.loc[img_id]["label"],
            }
        elif img_id in gt_test_unseen.index:
            retrieve_test_unseen[img_id] = {
                "text": gt_test_unseen.loc[img_id]["text"]
                + " "
                + " ".join(attobject_list),
                "label": gt_test_unseen.loc[img_id]["label"],
            }
    return retrieve_train, retrieve_val, retrieve_test_seen, retrieve_test_unseen


# This function is used for sparse_baseline experiment, sparse_baseline.py
# where training data is used for retrieval.
# For each query in the validation set, we find the top k most similar feature in the training set.
def sparse_retrieval(retrieve_train, retrieve_val, retrieve_size=30):
    """

    Args:
        retrieve_train (dictionary of dictionary): training data for retrieval
        retrieve_val (dictionary of dictionary): validation data for retrieval
        retrieve_size (int, optional): topk for retrieval. Defaults to 30.

    Returns:
        _type_: logging_dict that contains the retrieval results
    """

    # Form the corpus with the training text
    corpus = [retrieve_train[img_id]["text"] for img_id in retrieve_train]
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # Initialize the empty dic for logging
    logging_dict = EasyDict()

    for id in tqdm(retrieve_val):
        # Get the query text
        query = retrieve_val[id]["text"]
        tokenized_query = query.split(" ")
        # Find the best mathcing documents
        doc_scores = bm25.get_scores(tokenized_query)

        # Get the largest scores indices
        topk_indices = np.argsort(doc_scores)[::-1][:retrieve_size]
        # Get the image ids from the indices
        topk_ids = [list(retrieve_train.keys())[index]
                    for index in topk_indices]
        # Get the labels
        topk_labels = [retrieve_train[img_id]["label"] for img_id in topk_ids]
        # Get the retrieved scores
        retrieved_scores = [doc_scores[index] for index in topk_indices]
        logging_dict[id] = {
            "no_retrieved": len(topk_ids),
            "retrieved_ids": topk_ids,
            "retrieved_scores": retrieved_scores,
            "retrieved_label": topk_labels,
        }
    return logging_dict

# This function is used for actual RAC: retrieval augmented classification experiment,
# rac_full_sparse4hardnegative.py
# For each query in the training set (batch) during the training process,
# we find the top k most similar feature in the training set, but with opposite labels (hard negatives).


def sparse_retrieve_hard_negatives(retrieve_train, query_ids, retrieve_size=None):
    """
    Args:
        retrieve_train (dictionary of dictionary): training (database) data for retrieval
        query_ids (list of strings): query data for retrieval, using ids as identifiers
        retrieve_size (int, optional): topk for retrieval. Defaults to None,
                                        which means using all the training data. 

    Returns:
        _type_: logging_dict that contains the retrieval results
    """

    # Form the corpus with the training text
    corpus = [retrieve_train[img_id]["text"] for img_id in retrieve_train]
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # Initialize the empty dic for logging
    logging_dict = EasyDict()

    for id in tqdm(query_ids):
        # Get the query text
        query = retrieve_train[id]["text"]
        label = retrieve_train[id]["label"]
        tokenized_query = query.split(" ")
        # Find the best mathcing documents
        doc_scores = bm25.get_scores(tokenized_query)

        # Get the scores indices in descending order
        # (The reference of the index is defined by the retrieve_train)
        all_indices = np.argsort(doc_scores)[::-1]
        # Get the image ids from the indices
        # write to a list, thus now we have a list of ids with descending scores
        all_ids = [list(retrieve_train.keys())[index] for index in all_indices]
        # Now that we have the ids, we can get the labels for each items
        all_labels = [retrieve_train[img_id]["label"] for img_id in all_ids]
        # Get the retrieved scores for weighting the loss later on
        retrieved_scores = [doc_scores[index] for index in all_indices]

        # Since we are using hard negatives, we need to remove the positive samples
        # For now use a for loop, but can be optimized later on
        """for i in range(len(all_labels)):
            if all_labels[i] == label:
                all_ids.pop(i)
                all_labels.pop(i)
                retrieved_scores.pop(i)
        """
        all_ids_new = []
        retrieved_scores_new = []
        for i in range(len(all_labels)):
            if all_labels[i] != label:
                all_ids_new.append(all_ids[i])
                retrieved_scores_new.append(retrieved_scores[i])

        # If we have retrieve_size, then we only keep the topk
        if retrieve_size is not None:
            topk_ids = all_ids_new[:retrieve_size]
            # topk_labels = all_labels[:retrieve_size]
            retrieved_scores = retrieved_scores_new[:retrieve_size]
        else:
            topk_ids = all_ids_new
            # topk_labels = all_labels
            retrieved_scores = retrieved_scores_new

        # assert len(topk_ids) == len(retrieved_scores)
        """for i in topk_labels:
            assert i != label"""

        logging_dict[id] = {
            "query_id": id,
            "retrieved_ids": topk_ids,
            "retrieved_scores": retrieved_scores,
            # "retrieved_label": topk_labels,
        }

    return logging_dict


def dense_retrieve_hard_negatives_pseudo_positive(
    train_dl, query_feats, query_labels, model,
    largest_retrieval=1, threshold=None, args=None,
    train_feats=None, train_labels=None, reindex=False, yes_token_id=None
):

    # Get the batch size, do not use args.batch_size,
    # since the last batch might be smaller
    batch_size = query_feats.shape[0]

    if args.Faiss_GPU == False:
        # For cpu implementation
        query_feats = query_feats.cpu().detach().numpy().astype("float32")

    # If we set the train_feats and train_labels to None in upper level,
    # We will reindex the searching index with updated training data
    if (train_feats == None or train_labels == None) or reindex:
        print("Start to reindex dense retrieval index")
        for i, batch in enumerate(train_dl):
            with torch.inference_mode():
                # Change the item in the batch to be correct dtype
                for key in batch:
                    if key == "images":
                        # Hard code the bfloat16 here
                        if args.bf16:
                            batch[key] = batch[key].to(torch.bfloat16)
                
                _, _, all_feats = model(
                    **batch, classification_mode=True, output_hidden_states=True, output_embeds=True)

            # Extract labels
            labels = batch.get('labels')

            # Determine target labels (0 for 'No', 1 for 'Yes')
            has_yes = (labels == yes_token_id).any(dim=1)
            target_labels = torch.zeros(labels.size(0), device=labels.device)
            target_labels[has_yes] = 1

            if i == 0:

                if args.Faiss_GPU:
                    # For GPU implementation
                    train_feats = all_feats
                    train_labels = target_labels
                else:
                    # For cpu implementation

                    train_feats = all_feats.cpu().detach().numpy().astype("float32")
                    train_labels = target_labels.cpu(
                    ).detach().numpy().astype("int")

            else:

                if args.Faiss_GPU:
                    # For GPU implementation
                    train_feats = torch.cat((train_feats, all_feats), dim=0)
                    train_labels = torch.cat(
                        (train_labels, target_labels), dim=0)
                else:
                    # For cpu implementation
                    train_feats = np.concatenate(
                        (train_feats, all_feats.cpu().detach().numpy().astype("float32")))
                    train_labels = np.concatenate(
                        (train_labels, target_labels.cpu().detach().numpy().astype("int")))

    # Perform dense retrieval
    # Get the dimension of the features
    dim = train_feats.shape[1]
    # Initialize the index
    # For different loss functions, we need to change the index type
    if args.rgcl_metrics == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        index = faiss.IndexFlatIP(dim)

    # print("start to add the training features to the index")
    if args.Faiss_GPU:
        # print("start to transfer the FAISS index to GPU")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

        if args.rgcl_metrics != "ip":

            train_feats_normalized = torch.nn.functional.normalize(
                train_feats, p=2, dim=1)
            query_feats_normalized = torch.nn.functional.normalize(
                query_feats, p=2, dim=1)
        else:
            train_feats_normalized = train_feats
            query_feats_normalized = query_feats

    else:
        if args.rgcl_metrics != "ip":
            train_feats_normalized = train_feats
            query_feats_normalized = query_feats
            faiss.normalize_L2(train_feats_normalized)
            faiss.normalize_L2(query_feats_normalized)
        else:
            train_feats_normalized = train_feats
            query_feats_normalized = query_feats

    index.add(train_feats_normalized.to(torch.float32))

    # Search at most args.hardest_negatives_multiple of the largest retrieval no.

    D, I = index.search(query_feats_normalized.to(torch.float32), 
                        largest_retrieval*args.hard_negatives_multiple)

    # Initialize the hard negative features
    # Each item in the batch has no_hard_negatives hard negatives
    # Thus, this is a 3D tensor of size batch_size x no_hard_negatives x dim
    hard_negative_features = torch.zeros(
        batch_size, args.no_hard_negatives, dim, device="cuda", dtype=query_feats.dtype)

    if args.no_pseudo_gold_positives != 0:
        pseudo_positive_features = torch.zeros(
            batch_size, args.no_pseudo_gold_positives, dim, device="cuda", dtype=query_feats.dtype)
    # hard_negative_features = query_feats.unsqueeze(1).expand(batch_size,largest_retrieval, -1)

    # Initialize the hard negative retrieved scores
    hard_negative_scores = torch.zeros(
        batch_size, largest_retrieval, device="cuda", dtype=query_feats.dtype)
    if args.no_pseudo_gold_positives != 0:
        pseudo_positive_scores = torch.zeros(
            batch_size, args.no_pseudo_gold_positives, device="cuda", dtype=query_feats.dtype)

    # Fill the hard_negative_features with the original features multiplied by -1
    # so that in case, we did not find enough hard negatives, we can still use the original features
    # to compute the cos similarity loss.
    # If we do not do this, then the loss will be undefined
    # By doing so, we will get a cosine similarity of -1, which is the perfect dissimilar score,
    # and we thus minimizes the loss (giving some sort of reward in not finding the hard negatives)
    #

    for i, row in enumerate(D):

        # Initialize the counter for the number of hard negatives
        j = 0

        # initalize the counter for the number of pseudo gold positives
        k = 0
        for iter, value in enumerate(row):
            # print(query_labels[i].item())
            # If the label is opposite (negative)
            # print(train_labels[I[i, j]].item(), query_labels[i].item(), query_labels[i], train_labels[I[i, j]].item() != query_labels[i].item())

            # For the hard negatives
            if train_labels[I[i, iter]].item() != query_labels[i].item() and j < args.no_hard_negatives:

                if args.Faiss_GPU:
                    # GPU implementation
                    hard_negative_features[i][j] = train_feats[I[i, iter]]
                    hard_negative_scores[i][j] = value
                else:

                    # CPU implementation with numpy
                    hard_negative_features[i][j] = torch.from_numpy(
                        train_feats[I[i, iter]]).to("cuda")
                    hard_negative_scores[i][j] = torch.from_numpy(
                        np.asarray(value)).to("cuda")

                j += 1

            # For the pseudo gold positives
            elif train_labels[I[i, iter]].item() == query_labels[i].item() and k < args.no_pseudo_gold_positives:
                if args.Faiss_GPU:
                    # GPU implementation
                    pseudo_positive_features[i][k] = train_feats[I[i, iter]]
                    pseudo_positive_scores[i][k] = value
                else:
                    # CPU implementation with numpy
                    pseudo_positive_features[i][k] = torch.from_numpy(
                        train_feats[I[i, iter]]).to("cuda")
                    pseudo_positive_scores[i][k] = torch.from_numpy(
                        np.asarray(value)).to("cuda")

                k += 1
            # Only if both the number of hard negatives and pseudo gold positives are found, then break
            if j == largest_retrieval and k == args.no_pseudo_gold_positives:
                break
        #print("Searched top {} to get {} hard negatives".format(iter+1, j))

    if args.no_pseudo_gold_positives == 0:
        return hard_negative_features, hard_negative_scores, train_feats, train_labels
    elif args.no_pseudo_gold_positives != 0:
        return hard_negative_features, hard_negative_scores, pseudo_positive_features, pseudo_positive_scores, train_feats, train_labels


def sparse_retrieve_hard_negatives_pseudo_positive(
    ids,
    labels,
    train_set,
    model,
    sparse_retrieval_dictionary,
    args,
):

    all_ids = train_set.ids
    # Give an id as key,
    # the dictionary gives you the index in the trainset

    # TODO find new ways to do this
    # ids2index = train_set.ids_dics
    ids2index = {k: v for v, k in enumerate(all_ids)}

    # Train_len * feats
    all_img_features = train_set.image_feats
    all_text_features = train_set.text_feats
    all_labels = train_set.labels
    batch_size = len(ids)
    hard_negative_features = torch.zeros(
        batch_size, args.no_hard_negatives, args.proj_dim, device="cuda"
    )

    pseudo_positive_features = torch.zeros(
        batch_size, args.no_pseudo_gold_positives, args.proj_dim, device="cuda")
    hard_positive_features = torch.zeros(
        batch_size, args.no_hard_positives, args.proj_dim, device="cuda")

    # Interate over all the examples in the batch
    for index_batch, (idx, labelx) in enumerate(zip(ids, labels)):

        # For a sample in the batch,
        # we retrieve the top k most similar feature in the training set
        # K is defined when generating the sparse_retrieval_dictionary

        # Get the retrieved ids and labels for the sample in the batch
        retrieved_id_list = sparse_retrieval_dictionary[idx]["retrieved_ids"]
        retrieved_label_list = sparse_retrieval_dictionary[idx]["retrieved_labels"]

        # Initialize the counter for the number of hard/pseudo
        pseudo_positive_counter = 0
        hard_negative_counter = 0
        if args.sparse_topk == None or args.sparse_topk == -1:
            args.sparse_topk = len(retrieved_id_list)
        for index_topk, (retrieved_id, retrieved_label) in enumerate(zip(retrieved_id_list, retrieved_label_list)):
            # Get the index in the train set to get the features
            index_trainset = ids2index[retrieved_id]
            # Check if the index is correct by checking the label and ids
            assert all_labels[index_trainset] == retrieved_label, \
                "Sparse retrieval label mismatch"
            assert all_ids[index_trainset] == retrieved_id, \
                "Sparse retrieval id mismatch"
            # Encode the features with the model
            # Unsqueeze to add the batch dimension
            model.eval()

            # When same label, psuedo gold
            if retrieved_label == labelx and pseudo_positive_counter < args.no_pseudo_gold_positives:
                _, encoded_feature_x = model(
                    all_img_features[index_trainset].to(
                        args.device).unsqueeze(0),
                    all_text_features[index_trainset].to(
                        args.device).unsqueeze(0),
                    return_embed=True
                )
                pseudo_positive_features[index_batch][pseudo_positive_counter] = encoded_feature_x
                pseudo_positive_counter += 1
            # When different label, hard negative
            elif retrieved_label != labelx and hard_negative_counter < args.no_hard_negatives:
                _, encoded_feature_x = model(
                    all_img_features[index_trainset].to(
                        args.device).unsqueeze(0),
                    all_text_features[index_trainset].to(
                        args.device).unsqueeze(0),
                    return_embed=True
                )
                hard_negative_features[index_batch][hard_negative_counter] = encoded_feature_x
                hard_negative_counter += 1
            # If both hard negatives and pseudo gold positives are found,
            # then break the inside loop
            elif pseudo_positive_counter >= args.no_pseudo_gold_positives and hard_negative_counter >= args.no_hard_negatives:
                break
            elif index_topk >= args.sparse_topk:
                break
        # If not enough hard negatives are found, then just keep zero in the feature space
    return hard_negative_features, pseudo_positive_features
