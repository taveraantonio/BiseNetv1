import os
import argparse
import numpy as np
import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from model.build_BiSeNet import BiSeNet
from dataset import dataset
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, \
    per_class_iu
from loss import DiceLoss
import wandb
import utils

# wrapper for logging masks to W&B
def wb_mask(bg_img, pred_mask=[], true_mask=[]):
  masks = {}
  if len(pred_mask) > 0:
    masks["prediction"] = {"mask_data" : pred_mask}
  if len(true_mask) > 0:
    masks["ground truth"] = {"mask_data" : true_mask}
  return wandb.Image(bg_img, masks=masks, 
    classes=wandb.Classes([{'name': name, 'id': id} 
      for name, id in zip(utils.CLASSES, utils.IDS)]))

def val(args, model, dataloader, final_test):
    # init wandb artifacts
    # save validation predictions, create a new version of the artifact for each epoch
    val_res_at = wandb.Artifact("val_pred_" + wandb.run.id, "val_epoch_preds")
    # store all final results in a single artifact across experiments and
    # model variants to easily compare predictions
    final_model_res_at = wandb.Artifact("bisenet_pred", "model_preds")
    main_columns = ["prediction", "ground_truth"]
    # we'll track the IOU for each class
    main_columns.extend(["iou_" + s for s in utils.CLASSES])
    # create tables
    val_table = wandb.Table(columns=main_columns)
    model_res_table = wandb.Table(columns=main_columns)
    
    print('start val!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data
            label = label.long()

            # get RGB predict image
            predict = model(data).squeeze()
            predict = predict.squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            current_hist = fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            hist += current_hist

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)
            
            # add row to the wandb table
            row = [wb_mask(data, pred_mask=predict), wb_mask(data, true_mask=label)]
            row.extend(per_class_iu(current_hist))
            val_table.add_data(*row)
            if final_test == True:
                model_res_table.add_data(*row)
        
        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print(f'precision per pixel for test: {precision:.3f}')
        print(f'mIoU for validation: {miou:.3f}')
        print(f'mIoU per class: {miou_list}')
        
        # upload wandb table
        val_res_at.add(val_table, "val_epoch_results")
        wandb.run.log_artifact(val_res_at)
        if final_test == True:
            final_model_res_at.add(model_res_table, "model_results")
            wandb.run.log_artifact(final_model_res_at)

        return precision, miou


def train(args, model, optimizer, dataloader_train, dataloader_val):
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))
    scaler = amp.GradScaler()
    if args.loss == 'dice':
        loss_func = DiceLoss()
    elif args.loss == 'crossentropy':
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    max_miou = 0
    step = 0
    for epoch in range(args.init_epoch, args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i, (data, label) in enumerate(dataloader_train):
            data = data
            label = label.long()
            optimizer.zero_grad()
            
            with amp.autocast():
                output, output_sup1, output_sup2 = model(data)
                loss1 = loss_func(output, label)
                loss2 = loss_func(output_sup1, label)
                loss3 = loss_func(output_sup2, label)
                loss = loss1 + loss2 + loss3
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        wandb.log({"loss": loss_train_mean})
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        is_right_iteration = lambda i, step: (i % step) == (step - 1)
        if is_right_iteration(epoch, args.checkpoint_step) and args.model_file_name is not None:
            model_path = os.path.join(args.saved_models_path, args.model_file_name)
            torch.save(model.module.state_dict(), model_path) 

        if is_right_iteration(epoch, args.validation_step) and args.model_file_name is not None:
            precision, miou = val(args, model, dataloader_val, False)
            if miou > max_miou:
                max_miou = miou
                best_model_path = os.path.join(args.saved_models_path, 'best_'+args.model_file_name)
                torch.save(model.module.state_dict(), best_model_path) 
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou_val', miou, epoch)


def main():
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--init_epoch', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="Cityscapes", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=1024, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=1, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--model_file_name', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--saved_models_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='crossentropy', help='loss function, dice or crossentropy')

    args = parser.parse_args()


    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = torch.nn.DataParallel(model, output_device = device)
    
    # Create datasets instance
    dataset_path = os.path.join(args.data, args.dataset)
    new_size = (args.crop_height, args.crop_width)
    dataset_train = dataset.SegmentationDataset(dataset_path, new_size, 'train', device)
    dataset_val = dataset.SegmentationDataset(dataset_path, new_size, 'val', device)

    # Define your dataloaders:
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True)

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    # load pretrained model if exists
    if args.saved_models_path is not None:
        os.makedirs(args.saved_models_path, exist_ok=True)
    if args.model_file_name is not None:
        assert args.saved_models_path is not None, 'If you specify model file name, you must specify a directory'
        model_path = os.path.join(args.saved_models_path, args.model_file_name)
        if os.path.exists(model_path):
            print(f'Loading state from file: {model_path}')            
            model.module.load_state_dict(torch.load(model_path))
        else:
            print(f'Could not find model {model_path}, starting from zero')
    else:
        print('[WARNING] Model name was not specified. No data will be stored after training')

    # wandb logging init
    wandb.login()
    run = wandb.init(project=utils.WANDB_PROJECT, entity=utils.WANDB_ENTITY, job_type="train", config=args)
    wandb.watch(model, log_freq=args.batch_size)
    
    # train
    train(args, model, optimizer, dataloader_train, dataloader_val)
    # final test
    val(args, model, dataloader_val, True)
    
    # save model in wandb and close connection
    if args.saved_models_path is not None and args.model_file_name is not None:
        best_model_path = os.path.join(args.saved_models_path, 'best_'+args.model_file_name)
        if os.path.exists(best_model_path ):
            model_name = args.model_file_name
            saved_model = wandb.Artifact(model_name, type="model")
            saved_model.add_file(save_path, name=model_name)
            print("Saving data to WandB...")
            run.log_artifact(saved_model)
    run.finish()
    print("Completed run")


if __name__ == '__main__':
    main()
