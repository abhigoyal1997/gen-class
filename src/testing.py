import torch

from torch.utils.data import DataLoader


def test(model, dataset, model_path=None, device=torch.device('cpu'), predict_only=False):
    batch_size = 32
    num_workers = 4

    # TODO: create batches of data without masks
    batches = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    # Test
    metrics = model.predict(batches, device=device)
    predictions = metrics.pop('predictions', None)  # TODO: save the predictions, maybe?
    if predict_only:
        return predictions
    print('Test: {}'.format(metrics))
