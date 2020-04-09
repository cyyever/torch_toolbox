import itertools
import torch
import numpy as np


class SimilarityMetric:
    def __init__(self):
        pass

    def compare(self, s1, s2):
        """ 检查s1的相似度是否比s2更高 """
        return s1 < s2


class MinkowskiDistance(SimilarityMetric):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def get_metric(self, x, y):
        return torch.nn.functional.pdist(torch.stack([x, y]), p=self.p)

    def get_pair_wise_metric(self, samples):
        return torch.nn.functional.pdist(samples, p=self.p)


class Cluster:
    def __init__(self, similarity_metric):
        self.similarity_metric = similarity_metric
        self.samples = set()

    def add_sample(self, sample):
        self.samples.add(sample)

    def merge(self, other_cluster):
        self.samples |= other_cluster.samples

    def remove_sample(self, sample):
        self.samples.discard(sample)

    def has_sample(self, sample):
        return sample in self.samples

    def get_metric_from_center(self, sample):
        return self.similarity_metric.get_metric(self.get_center(), sample)

    def get_center(self):
        return torch.Tensor(list(self.samples)).mean(dim=0)

    def get_diameter(self):
        return self.__get_maxmin_metric(
            itertools.combinations(
                self.samples, 2), False)

    def __get_maxmin_metric(self, iterable, want_most_similar):
        result = None
        for x, y in iterable:
            metric = self.similarity_metric.get_metric(x, y)
            if result is None:
                result = metric
            more_similar = self.similarity_metric.compare(metric, result)
            if (more_similar and want_most_similar) or (
                not more_similar and not want_most_similar
            ):
                result = metric
        return result

    def get_single_linkage(self, other_cluster):
        return self.__get_maxmin_metric(
            itertools.product(self.samples, other_cluster.samples), True
        )

    def get_complete_linkage(self, other_cluster):
        return self.__get_maxmin_metric(
            itertools.product(self.samples, other_cluster.samples), False
        )

    def get_average_linkage(self, other_cluster):
        total_metric = 0
        for x, y in itertools.product(self.samples, other_cluster.samples):
            total_metric += self.similarity_metric.get_metric(x, y)
        return total_metric / (len(self.samples) * len(other_cluster.samples))

    def get_center_linkage(self, other_cluster):
        return self.similarity_metric.get_metric(
            self.get_center(), other_cluster.get_center()
        )


class HierarchicalClustering:
    def __init__(self, samples, similarity_metric=MinkowskiDistance(p=2)):
        self.samples = samples
        self.similarity_metric = similarity_metric

    def agglomerative(self, stop_condition):
        """ 实现聚合层次聚类 """
        # 首先每个样本各自一个群
        clusters = []
        for sample in self.samples:
            cluster = Cluster(self.similarity_metric)
            cluster.add_sample(sample)
            clusters.append(cluster)
        while not stop_condition(clusters):
            most_similarity = None
            merged_indices = None

            for i, j in itertools.combinations(range(len(clusters)), 2):
                similarity = clusters[i].get_center_linkage(clusters[j])
                if most_similarity is None or self.similarity_metric.compare(
                    similarity, most_similarity
                ):
                    most_similarity = similarity
                    merged_indices = (i, j)

            i, j = merged_indices
            clusters[i].merge(clusters[j])
            del clusters[j]
        return clusters


class KMeans:
    def __init__(self, samples, k):
        self.samples = samples
        if k > len(self.samples) or k == 1:
            raise RuntimeError("invalid k ")
        self.k = k

    def compute(self):
        # 首先随机选择k个样本作为cluster中心
        clusters = []
        for index in np.random.choice(
                len(self.samples), self.k, replace=False):
            cluster = Cluster(MinkowskiDistance())
            cluster.add_sample(self.samples[index])
            clusters.append(cluster)

        while True:
            stop = True
            new_clusters = [Cluster(MinkowskiDistance())
                            for _ in range(self.k)]
            for sample in self.samples:
                index_min = np.argmin(
                    [c.get_metric_from_center(sample) for c in clusters]
                )
                new_clusters[index_min].add_sample(sample)
                if not clusters[index_min].has_sample(sample):
                    stop = False
            clusters = new_clusters
            if stop:
                break
        return clusters
