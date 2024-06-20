import os
import sys
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dassl.optim.lr_scheduler import ConstantWarmupScheduler
from decoder_model import DecoderModel
from decoder_dataset import DecoderDataset


MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
WARMUP_LR_RATIO = 2e-3
NUM_EPOCHS = 50
NUM_CLASSES_DICT = assert(False) #KEVIN
NUM_WORKERS = assert(False) #KEVIN
OUT_PARENT_DIR = assert(False) #KEVIN


#HOOK
def make_params(input_type, num_hidden_layers, hidden_layer_size, use_dropout, use_batchnorm, lr):
    assert(False) #KEVIN


#return model_filename, result_dict_filename
#will create directory if it's not already created
def make_out_filenames(dataset_name, params, epoch):
    p = params
    p_str = 'decoder_%s_%s_nhl%d_hls%d_dropout%d_batchnorm%d_lr%s_epoch%d'%(dataset_name.split('_')[0], p['input_type'], p['num_hidden_layers'], p['use_intermediate_dropout'], p['use_intermediate_batchnorm'], str(p['lr']), epoch+1)
    out_dir = os.path.join(OUT_PARENT_DIR, p_str)
    os.makedirs(out_dir, exist_ok=True)
    model_filename = os.path.join(out_dir, p_str + '_model.pth')
    result_dict_filename = os.path.join(out_dir, p_str + '_result_dict.pth')
    return model_filename, result_dict_filename


#return optimizer, scheduler
def make_optimizer_and_scheduler(model, params):
    p = params
    optimizer = torch.optim.SGD(model.parameters(), lr=p['lr'], momentum=p['momentum'], weight_decay=p['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, p['num_epochs'])
    scheduler = ConstantWarmupScheduler(optimizer, scheduler, 1, p['warmup_lr'])
    return optimizer, scheduler


#use same scheduler as DC, but with different lr and warmup_lr
#if dropout/batchnorm is used, it's used everywhere. dropout_prob is fixed
#warmup_lr is always fixed ratio of lr
#always use standardization
#we're not even implementing residuals, because the input type might not be logits
#number of epochs is fixed, but we test after every epoch
def train_decoder(dataset_name, input_type, num_hidden_layers, hidden_layer_size, use_dropout, use_batchnorm, lr):
    num_hidden_layers = int(num_hidden_layers)
    hidden_layer_size = int(hidden_layer_size)
    use_dropout = int(use_dropout)
    use_batchnorm = int(use_batchnorm)
    lr = float(lr)

    p = make_params(input_type, num_hidden_layers, hidden_layer_size, use_dropout, use_batchnorm, lr)
    train_dataset = DecoderDataset(dataset_name, 'train', p)
    test_dataset = DecoderDataset(dataset_name, 'test', p)
    train_dataloader = DataLoader(train_dataset, batch_size=p['batch_size'], num_workers=NUM_WORKERS, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=p['batch_size'], num_workers=NUM_WORKERS, shuffle=False, drop_last=False)
    
    #standardization
    print('computing standardization...')
    train_scores = []
    for batch in tqdm(train_dataloader):
        train_scores.append(batch['scores'])

    with torch.no_grad():
        train_scores = torch.cat(train_scores, dim=0)
        train_scores = train_scores.cuda()
        standardization_info = {'means' : torch.mean(train_scores, dim=0), 'sds' : torch.std(train_scores, dim=0)}

    print('done computing standardization')

    model = DecoderModel(NUM_CLASSES_DICT[dataset_name], standardization_info, p)
    optimizer, scheduler = make_optimizer_and_scheduler(model, p)
    for epoch in range(p['num_epochs']):
        
        #training
        loss_buffer = []
        model.train()
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            scores = batch['scores'].cuda()
            gts = batch['gts'].cuda()
            logits = model(scores)
            losses = F.binary_cross_entropy_with_logits(logits, gts)
            loss = torch.mean(torch.sum(losses, dim=1))
            loss.backward()
            loss_buffer.append(loss.detach().item())
            if len(loss_buffer) % PRINT_FREQ == 0:
                print('epoch=%d, steps=%d, recent_avg_loss=%f'%(epoch+1, len(loss_buffer), np.mean(loss_buffer[-PRINT_FREQ:])))
            optimizer.step()
        
        scheduler.step()
        
        #evaluation
        model.eval()
        test_logits = []
        test_gts = []
        test_impaths = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                scores = batch['scores'].cuda()
                gts = batch['gts']
                logits = model(scores)
                test_logits.append(logits)
                test_gts.append(gts)
                impaths = test_dataset.get_impaths(batch['idx'])
                test_impaths.extend(impaths)

            test_logits = torch.cat(test_logits, dim=0)
            test_gts = torch.cat(test_gts, dim=0)
            test_logits = test_logits.cpu().numpy()
            test_gts = test_gts.cpu().numpy()

        class_APs = average_precision(test_logits, test_gts)
        mAP = np.mean(class_APs)

        #print
        print('Epoch %d: mAP=%f'%(epoch+1, mAP))

        #result dict
        result_dict = {'dataset_name' : dataset_name, 'params' : p, 'epoch' : epoch+1, 'mAP' : mAP, 'class_APs' : class_APs, 'test_logits' : test_logits, 'test_impaths' : test_impaths}

        #saving
        model_filename, result_dict_filename = make_out_filenames(dataset_name, p, epoch)
        torch.save(result_dict, result_dict_filename)
        torch.save({'model_state_dict' : model.state_dict(), 'optimizer_state_dict' : optimizer.state_dict(), 'scheduler_state_dict' : scheduler.state_dict(), 'epoch' : epoch+1})


def usage():
    print('Usage: python train_decoder.py <dataset_name> <input_type> <num_hidden_layers> <hidden_layer_size> <use_dropout> <use_batchnorm> <lr>')


if __name__ == '__main__':
    train_decoder(*(sys.argv[1:]))
