# 模型预测
import torch
import cv2
from visualize import *
from torch.utils.data import DataLoader, Dataset
from torch import nn
from train_ import BalloonDataset, annotations_test, testset_dir, train_transform, train_loader
from train_big import UNetModel
import torchmetrics

device = torch.device("cuda")
test_dataset = BalloonDataset(annotations_test, testset_dir, transform=train_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#
model = UNetModel(128,128,3).to(device)
model.load_state_dict(torch.load("models/G_big_lr1e-4/best_model_300.pth"))


def predict(model, test_loader, device):
    model.eval()
    criterion = nn.BCELoss().to(device)
    val_loss = 0.0
    val_iou = 0.0
    i = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.bool()
            masks = masks.float()
            outputs = model(images)
            loss = criterion(outputs, masks)  # 计算验证集损失
            val_loss += loss.item() * images.size(0)

            thres = (torch.max(outputs) - torch.min(outputs)) * 1 / 2 + torch.min(outputs)
            bool = outputs[0] > thres
            bool = bool.float()
            iou = torchmetrics.functional.classification.binary_jaccard_index(outputs, masks,
                                                                              thres.item())  # iou_metric(outputs, masks, thres)# compute_iou(outputs, masks)
            val_iou += iou.item() * images.size(0)

            # 计算测试集的loss和ioU
            bool = bool.float()
            bool = bool * 255
            bool = bool.permute(1, 2, 0)
            image = masks[0].permute(1, 2, 0) * 255
            print(f"Image {i} - Loss: {loss.item() * images.size(0):.4f}, IoU: {iou.item() * images.size(0):.4f}")

            # 显示图片
            plot_image(images, image, bool)
            cv2.imwrite(f"output/bool_{i}.jpg", bool.cpu().numpy())
            cv2.imwrite(f"output/image_{i}.jpg", image.cpu().numpy())
            i += 1

    val_loss /= len(test_loader.dataset)
    val_iou /= len(test_loader.dataset)

    print(f'Val Loss: {val_loss:.4f}, '
          f'Val IoU: {val_iou:.4f}')


def plot_image(image, mask, pred):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    image = image.squeeze()
    image = image.permute(1, 2, 0)
    plt.imshow(image.cpu().detach().numpy())
    plt.title("Image")
    plt.subplot(1, 3, 2)
    plt.imshow(mask.cpu().detach().numpy().squeeze())
    plt.title("Ground Truth")
    plt.subplot(1, 3, 3)
    plt.imshow(pred.cpu().detach().numpy().squeeze(), cmap="gray")
    plt.title("Prediction")
    plt.show()


# def calculate_metrics(preds, masks, thres,  device):
#     # 计算loss
#     criterion = nn.BCELoss().to(device)
#     loss = criterion(preds, masks)
#     masks = masks.clamp(0, 1)
#     # 计算IoU - 直接使用预测值，不需要再次sigmoid
#     pred_binary = (preds > thres).float()
#     intersection = (pred_binary * masks).sum()
#     union = (pred_binary + masks).clamp(0, 1).sum()
#     iou = (intersection + 1e-6) / (union + 1e-6)
#     return loss.item(), iou.item()


# def save_predictions(preds, save_dir='pred_masks'):
#     os.makedirs(save_dir, exist_ok=True)
#     pred_binary = (torch.sigmoid(preds) > 0.5).float()
#
#     for i in range(len(pred_binary)):
#         pred_mask = pred_binary[i].squeeze().cpu().numpy()
#         plt.imsave(f'{save_dir}/pred_{i}.png', pred_mask, cmap='gray')
#

# def plot_predictions(test_images, test_masks, preds_test):
#     # 确保数据类型是torch.Tensor
#     if isinstance(test_images, np.ndarray):
#         test_images = torch.from_numpy(test_images)
#     if isinstance(test_masks, np.ndarray):
#         test_masks = torch.from_numpy(test_masks)
#     # 将预测结果连接成一个tensor
#     thres_ls = preds_test[:, 1]
#     preds_test = preds_test[:, 0]
#     preds = torch.cat(preds_test, dim=0)
#     preds = preds.to(device)
#     test_masks = test_masks.clamp(0, 1)
#     # 计算整体的平均loss和IoU
#     # avg_loss, avg_iou = calculate_metrics(preds, test_masks, device)
#     # print(f"Average Loss: {avg_loss:.4f}, Average IoU: {avg_iou:.4f}")
#
#     # 可视化每张图片并计算单张图片的指标
#     for i in range(len(test_images)):
#         image = test_images[i].permute(1, 2, 0)
#         mask = test_masks[i].squeeze()
#         pred = preds[i:i + 1]  # 保持维度以便计算loss
#         thres = thres_ls[i]
#
#         # 计算单张图片的loss和IoU
#         individual_loss, individual_iou = calculate_metrics(pred, test_masks[i:i + 1], thres, device)
#         print(f"Image {i} - Loss: {individual_loss:.4f}, IoU: {individual_iou:.4f}")
#
#         # 显示图片
#         pred_display = (pred > thres).float().squeeze()
#         plot_image(image, mask, pred_display)
#
#     save_predictions(preds)
#     # return avg_loss, avg_iou

preds_test = predict(model, test_loader, device)
##### need to be done ######
# 补全predict，保存预测掩模
#
# test_images, test_masks = next(iter(test_loader))
# plot_predictions(test_images.to(device), test_masks.to(device), preds_test)
##### need to be done ######
# 附加： 补全plot_predictions
# 计算平均loss和ioU，并保存预测研磨图片