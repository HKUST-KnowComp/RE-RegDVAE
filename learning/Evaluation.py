import numpy as np
import logging
import tensorflow as tf
from collections import defaultdict
from definitions import settings
from definitions import log_dict

# Atom Heart Mother


class ClusterEvaluator:
    def __init__(self, rel_predictions: np.array, entropy, rel_labels, step, epoch, key, rel_num):

        self.step = step
        self.epoch = epoch
        self.key = key

        self.rel_num = rel_num

        self.rel_predictions = rel_predictions
        self.raw_entropy = entropy

        try:
            self.rel_labels = rel_labels.astype(int)
            self.full_labled = True  # all data has been labeled
            self.pred_cluster_number = np.max(self.rel_predictions) + 1
            self.label_cluster_number = np.max(self.rel_labels) + 1
        except TypeError:
            self.rel_labels = rel_labels
            self.full_labled = False  # not all data has been labeled

        self.clusters, self.clusters_size = None, None
        self.pred_mat, self.pred_cluster_size = None, None
        self.joint_count = None
        self.efficiency = None
        self.pred_normalized_entropy = None
        self.logger = logging.getLogger(__name__)

        # results
        self.sparsity, self.b3, self.purity, self.nmi, self.stanford_f, self.entropy = (None,) * 6

        self.tf_summary_value = []

    def cal_all(self):
        self.cal_sparsity()
        self.cal_normalized_probability_entropy()
        self.cal_cluster_entropy()
        if self.full_labled:
            self.cal_b3()
            self.cal_purity()
            self.cal_nmi()
            self.cal_stanford_F()

    def cal_sparsity(self):
        if self.sparsity is None:
            rel_num = self.rel_num
            covered_num = len(set(self.rel_predictions))
            sparsity = 1 - covered_num / rel_num
            self.logger.debug("Sparsity: {:.5f}".format(sparsity))
            self._save_log("Sparsity", sparsity)
            self.sparsity = sparsity
        return self.sparsity

    def cal_normalized_probability_entropy(self):
        # https://en.wikipedia.org/wiki/Entropy_(information_theory)#Efficiency
        if self.efficiency is None:
            # the probability is along axis = 1
            # normalize is taken along axis = 1
            # average is taken along axis = 0
            efficiency = self.raw_entropy / len(self.rel_predictions) / np.log2(self.rel_num)
            self._save_log("AverageNormalizedProbabilityEntropy", efficiency)
            self.efficiency = efficiency
        return self.efficiency

    def cal_cluster_entropy(self):
        if self.pred_normalized_entropy is None:
            cluster_size = self._cal_pred_cluster_size()
            cluster_number = self.rel_num
            l = len(self.rel_predictions)

            ncs = cluster_size / l  # normalized cluster size
            ncs_log2 = np.nan_to_num(np.log2(ncs), False)

            normalized_entropy = -np.sum(ncs * ncs_log2) / np.log2(cluster_number)
            self._save_log("NormalizedClusterEntropy", normalized_entropy)

            self.pred_normalized_entropy = normalized_entropy

        return self.pred_normalized_entropy

    def _cal_pred_cluster_size(self):
        if self.pred_cluster_size is None:
            self.pred_cluster_size = np.bincount(self.rel_predictions)
        return self.pred_cluster_size

    @staticmethod
    def b3_dense_mat(rel_labels, rel_predictions):
        label_clusters, label_cluster_size = get_one_hot_cluster(rel_labels, )
        pred_clusters, pred_cluster_size = get_one_hot_cluster(rel_predictions)
        relation_prediction = rel_predictions
        intersection = np.logical_and(pred_clusters[relation_prediction], label_clusters[rel_labels])
        intersection_count = np.sum(intersection, axis=1)
        recall = intersection_count / label_cluster_size[rel_labels]
        precision = intersection_count / pred_cluster_size[relation_prediction]

        total_recall, total_precision = np.average(recall), np.average(precision)
        F1 = F_beta_score(total_precision, total_recall, 1)
        F0p5 = F_beta_score(total_precision, total_recall, 0.5)
        return total_precision, total_recall, F1, F0p5

    @staticmethod
    def b3_fast(label_clusters, label_cluster_size, pred_clusters, pred_cluster_size):

        out = np.matmul(label_clusters, pred_clusters.T, )
        out = out ** 2
        lc = label_cluster_size
        pc = pred_cluster_size
        recall = np.nan_to_num(np.sum(out, axis=1) / lc)
        precision = np.nan_to_num(np.sum(out, axis=0) / pc)

        l = label_clusters.shape[1]
        total_recall, total_precision = np.sum(recall) / l, np.sum(precision) / l
        F1 = F_beta_score(total_precision, total_recall, 1)
        F0p5 = F_beta_score(total_precision, total_recall, 0.5)
        return total_precision, total_recall, F1, F0p5

    def cal_b3(self):
        assert self.full_labled is True, "relation labels contains None"
        if self.b3 is not None:
            return self.b3
        total_precision, total_recall, F1, F0p5 = self.b3_fast(*self._get_cluster_onehot())
        self.logger.debug(
            "B3 F1 = {:.5f} F0.5 = {:.5f} Recall = {:.5f} Precision ={:.5f}".format(F1, F0p5, total_recall,
                                                                                    total_precision))

        self.b3 = settings.OieEvaluationResults(F1, F0p5, total_recall, total_precision)
        names = ('B3-F1', 'B3-F0.5', 'B3-Recall', 'B3-Precision')
        for name, value in zip(names, self.b3):
            self._save_log(name, value)
        return self.b3

    def cal_purity(self):
        assert self.full_labled is True, "relation labels contains None"
        if self.purity is not None:
            return self.purity
        # [pmn, lmn]
        su = self._get_joint_count()
        ma = np.max(su, axis=-1)
        correct_assigned = np.sum(ma)
        purity = correct_assigned / len(self.rel_predictions)
        self.logger.debug("Purity: {:.5f}".format(purity))
        self.purity = purity

        # log
        self._save_log("Purity", purity)
        return purity

    def cal_nmi(self):
        assert self.full_labled is True, "relation labels contains None"
        if self.nmi is None:
            # [pmn, lmn]
            # todo: use matmul
            joint_count = self._get_joint_count()

            mask = joint_count != 0
            masked_joint_count = joint_count[mask]

            l = len(self.rel_predictions)  # N
            _, label_cluster_size, _, pred_cluster_size = self._get_cluster_onehot()
            # outer product -> pmn, lmn
            outer = np.outer(pred_cluster_size, label_cluster_size)
            masked_outer = outer[mask]

            log = np.log2(l * masked_joint_count / masked_outer)
            log = np.nan_to_num(log, False)  # cast nan to 0, because joint count would also be 0
            mul = masked_joint_count * log / l
            mutual_information = np.sum(mul)

            pred_entropy = self._cal_entropy(pred_cluster_size, l)
            label_entropy = self._cal_entropy(label_cluster_size, l)

            NMI = mutual_information * 2 / (pred_entropy + label_entropy)

            self.logger.debug(
                "NMI: {:.5f} mutual_information: {:.5f} pred_entropy: {:.5f} label_entropy: {:.5f}".format(
                    NMI, mutual_information, pred_entropy, label_entropy))
            self.nmi = NMI
            self.entropy = pred_entropy

            self._save_log("NMI", NMI)
            # self._save_log("PredictEntropy", pred_entropy)

        return self.nmi, self.entropy

    def cal_stanford_F(self):
        assert self.full_labled is True, "relation labels contains None"
        if self.stanford_f is None:
            # generate table
            joint_count = self._get_joint_count()
            pred_cluster_size = np.sum(joint_count, axis=1)

            def cal_positive(ary):
                return np.sum(ary * (ary - 1)) / 2

            def cal_negative(ary):
                current_sum = np.sum(ary, axis=0)
                result = 0
                for i in range(len(ary)):
                    current_sum -= ary[i]
                    result += current_sum * ary[i]

                return np.sum(result)

            # TP+FP
            positive = cal_positive(pred_cluster_size)

            # TP
            tp = cal_positive(joint_count)
            fp = positive - tp

            # TN+FN
            negative = cal_negative(pred_cluster_size)

            # TN
            fn = cal_negative(joint_count)
            tn = negative - fn

            # F measure
            precision, recall = cal_precision_recall(tp, fp, tn, fn)
            f1 = F_beta_score(precision, recall)
            fhalf = F_beta_score(precision, recall, 0.5)

            self.logger.debug("Stanford precision: {:.5f} recall: {:.5f} f1: {:.5f} f0.5: {:.5f}".format(
                precision, recall, f1, fhalf
            ))

            self.stanford_f = settings.OieEvaluationResults(f1, fhalf, recall, precision)

            names = ["SF-F1", "SF-F0.5", "SF-Recall", "SF-Precision"]
            for name, value in zip(names, self.stanford_f):
                self._save_log(name, value)

        return self.stanford_f

    def _get_joint_count(self):
        if self.joint_count is None:
            # [pmn, 1, l]
            # [lmn, 1, l]
            lab_mat, _, pred_mat, _ = self._get_cluster_onehot()
            pred_mat = np.expand_dims(pred_mat, axis=1)
            lab_mat = np.expand_dims(lab_mat, axis=0)
            # [pmn, lmn, l]
            # [pmn, lmn]
            la = np.logical_and(pred_mat, lab_mat)
            self.joint_count = np.sum(la, axis=-1)
        return self.joint_count

    def _get_cluster_onehot(self):
        if self.clusters is None:
            self.clusters, self.clusters_size = get_one_hot_cluster(self.rel_labels, self.label_cluster_number)
            self.pred_mat, self.pred_cluster_size = get_one_hot_cluster(self.rel_predictions, self.pred_cluster_number)
        return self.clusters, self.clusters_size, self.pred_mat, self.pred_cluster_size

    def _save_log(self, eval_name, value):
        self.logger.info(log_dict.EvalLog(self.step, self.epoch, self.key,
                                          eval_name, value))
        self.tf_summary_value.append(
            tf.Summary.Value(tag="Evaluation/{}/{}".format(self.key, eval_name), simple_value=value, )
        )

    @staticmethod
    def _cal_entropy(cluster_size, l=0):
        if l == 0:
            l = np.sum(cluster_size)
        ncs = cluster_size / l  # normalized cluster size
        ncs_log2 = np.nan_to_num(np.log2(ncs), False)
        return -np.sum(ncs_log2 * ncs)

    # same as get_one_hot_cluster
    # @staticmethod
    # def _purity_mat(row, cluster_number):
    #     l = len(row)
    #     data = np.ones(l)
    #     col = np.arange(l)
    #     sp_mat = sparse.csr_matrix((data, (row, col)), shape=(cluster_number, l))
    #     return sp_mat.toarray()


def get_one_hot_cluster(labels, n=None):
    if n is None:
        n = np.max(labels) + 1
    clusters = np.eye(n)[labels].transpose()
    counts = np.sum(clusters, axis=1)
    return clusters, counts


def get_one_hot_sparse_cluster(labels, n=None):
    if n is None:
        n = np.max(labels) + 1


def F_beta_score(p, r, beta=1):
    beta_square = beta ** 2
    return (1 + beta_square) * (p * r) / (beta_square * p + r)


def cal_precision_recall(tp, fp, tn, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall


class LinkPredictionEvaluator:
    def __init__(self, data):
        self.logger = logging.getLogger(__name__)
        self.dataset_manager = data
        # use train set
        self.triplet_dict = self.get_triplet_dict(data.kb_triplets['train'])
        # data_row, raw/filtered, h/t/r
        self.current_eval_id = 0

        self.eval_names = [['entity', 'relation'],
                           ['raw', 'filtered'],
                           ['MRR', 'MR', 'Hits@1', 'Hits@3', 'Hits@10']]

    def new_evaluator(self):
        new = LinkPredictionEvaluator()
        new.triplet_dict = self.triplet_dict
        new.dataset = self.dataset
        return new

    def add_score(self, score_batchs, dataset):
        ranks = np.zeros((len(self.dataset_manager['train']), 2, 3))
        for predict_item, score_batch in enumerate(score_batchs):
            if score_batch is not None:
                for global_id, score_row in enumerate(score_batch):
                    triplet = dataset[global_id]
                    raw, filtered = self.kb_entity_link_prediction_get_rank(
                        score_row, triplet, predict_item)
                    ranks[global_id, :, predict_item] = (raw, filtered)
        return ranks

    def cal_result_by_rank(self, ranks, step, epoch, data_key, ):
        tf_summary_value = []
        eval_names = self.eval_names

        # q, 2, 3
        data_size = len(ranks)
        # e/r, raw/filtered,  MRR MR hits@1|3|10,
        results = np.zeros((2, 2, 5))
        # mean rank
        # 2, 3
        mean_rank_sum = ranks.sum(axis=0, ) / data_size
        self.update_results(results, mean_rank_sum, 1)

        # MRR
        rank_inverse = 1 / (ranks + 1)
        mrr_sum = np.sum(rank_inverse, axis=0, ) / data_size
        self.update_results(results, mrr_sum, 0)

        # hits@1,3,10
        for i, hits_n in enumerate((1, 3, 10)):
            hits = np.sum(ranks < hits_n, axis=0) / data_size
            self.update_results(results, hits, i + 2)

        for index, x in np.ndenumerate(results):
            name = "{}_{}_{}".format(
                eval_names[0][index[0]], eval_names[1][index[1]], eval_names[2][index[2]])
            self._save_log(tf_summary_value, name, x, step, epoch, data_key)
        return tf_summary_value, ranks

    def log_results(self, scores, step, epoch, data_key, ):
        dataset = self.dataset_manager.kb_triplets[data_key]
        ranks = self.add_score(scores, dataset)
        # to do: print 1-1 1-m m-1 m-m
        return self.cal_result_by_rank(ranks, step, epoch, data_key)

    def _save_log(self, tf_summary_value, eval_name, value, step, epoch, key):
        self.logger.info(log_dict.EvalLog(step, epoch, key,
                                          eval_name, value))
        tf_summary_value.append(
            tf.Summary.Value(tag="Evaluation/{}/{}".format(key, eval_name), simple_value=value, )
        )

    @staticmethod
    def update_results(results, eval_results, eval_id):
        results[1, :, eval_id] = eval_results[:, 2]
        results[0, :, eval_id] = (eval_results[:, 1] + eval_results[:, 0]) / 2

    @staticmethod
    def get_rank(score_list, target_value):
        return (score_list > target_value).sum()

    @staticmethod
    def get_triplet_dict(dataset):
        if dataset is None:
            return None
        # :return : list[ relnum , list[ 2 , dict{entity_i: list[entity_j] } ] ]
        # d[relation_id][tail(0) head(1)][entity_id] -> entity_id
        # list list dict list
        # d = [[defaultdict(list) for j in range(2)] for i in range(rel_num)]
        # for _, e1, e2, rl in dataset:
        #     d[rl][0][e1].append(e2)  # pred tail
        #     d[rl][1][e2].append(e1)  # pred head
        d = defaultdict(list)
        for _, e1, e2, rl in dataset:
            d[(-1, e2, rl)].append(e1)
            d[(e1, -1, rl)].append(e2)
            d[(e1, e2, -1)].append(rl)
        return d

    def kb_entity_link_prediction_get_rank(self, score_list, triplet, predict_item):
        triplet = triplet[1:]
        target = triplet[predict_item]
        target_value = score_list[target]
        lookup_key = triplet.copy()
        lookup_key[predict_item] = -1
        lookup_key = tuple(lookup_key)
        known_eids = self.triplet_dict[lookup_key]
        raw_rank = self.get_rank(score_list, target_value)
        score_list[known_eids] = -1e8
        filter_rank = self.get_rank(score_list, target_value)
        return raw_rank, filter_rank
