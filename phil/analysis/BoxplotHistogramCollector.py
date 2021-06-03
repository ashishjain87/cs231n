from typing import Tuple, List
import matplotlib.pyplot as plt


class BoxplotHistogramCollector:
    """
    An object that creates box plots representing conditional distributions of a targeted metric based on some
    condition score (like 'occlusion severity').
    """
    lower_bound: float
    upper_bound: float
    metric_instances: List[Tuple[float, float]]

    def __init__(self, lower_bound: float = 0.0, upper_bound: float = 1.0):
        """

        :param lower_bound: minimum possible value for the x axis
        :param upper_bound: maximum possible value for the x axis
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.metric_instances = []

    def record_event(self, metric: float, condition_score: float):
        """
        Call this to contribute a data point to the histogram

        :param metric: y axis value for this example
        :param condition_score: x axis value for this example
        :return:
        """
        self.metric_instances.append((condition_score, metric))

    def produce_histogram(self, num_bins: int, title: str, y_label: str, x_label: str, savepath: str):
        fig = plt.figure()

        bin_groups = [[] for i in range(num_bins)]
        for condition_score, metric in self.metric_instances:
            rank = condition_score - self.lower_bound
            bin = min(int(num_bins*rank/(self.upper_bound - self.lower_bound)), num_bins - 1)
            bin_groups[bin].append(metric)

        positions = [0.5 + i for i in range(num_bins)]
        plt.boxplot(x=bin_groups, positions=positions)
        labels = [str(round((self.upper_bound - self.lower_bound)/num_bins * i)) for i in range(num_bins + 1)]

        plt.xticks(ticks=range(num_bins + 1), labels=labels)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        fig.suptitle(title, fontsize=11)
        plt.savefig(savepath)


def test_this():
    h = BoxplotHistogramCollector(0, 10000)
    for i in range(10000):
        h.record_event(i ** 1.2, i)
    h.produce_histogram(10, "title", "y axis label", "x axis label", "~/save.png")


if __name__ == "__main__":
    # a demo! Change 'plt.savefig(savepath)' to 'plt.show()' to see an example output
    test_this()
