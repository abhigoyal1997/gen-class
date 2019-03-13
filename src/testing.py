from torch.utils.data import DataLoader


def test(model, dataset, model_path=None, predict=False):
    batch_size = 32
    num_workers = 4

    batches = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    metrics = model.predict(batches, 'label' in dataset.df, predict)
    # metrics = model.run_epoch('test', batches, criterion=model.get_criterion())
    if predict:
        predictions = metrics.pop('predictions', None)  # TODO: save the predictions, maybe?
        return predictions

    print('Test: {}'.format(metrics))
