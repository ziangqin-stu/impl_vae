import torch


def plot_image_plane(imgs, row, col):
    assert row * col <= imgs.shape[0]
    x = torch.zeros(1, 28 * col)
    for i in range(row):
        x_row = imgs[(i - 1) * col].reshape(28, 28)
        for j in range(1, col):
            x_row = torch.cat([x_row, imgs[(i - 1) * col + j].reshape(28, 28)], 1)
        x = torch.cat([x, x_row], 0)
    return x[1:]
