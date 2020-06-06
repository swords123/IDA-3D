from maskrcnn_benchmark.data import datasets

from .kitti import kitti_evaluation


def evaluate(dataset, predictions, output_folder, **kwargs):
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
    )
    if isinstance(dataset, datasets.KittiDataset):
        return kitti_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
