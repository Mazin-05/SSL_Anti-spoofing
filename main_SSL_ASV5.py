import argparse
import random
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
# --- ASVspoof 5 Imports ---
from data_utils_ASV5 import genSpoof_list_ASV5, Dataset_ASVspoof5

# --- In-Memory Metric Engine Imports ---
from sklearn.metrics import roc_curve, accuracy_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from collections import defaultdict
import warnings

# Suppress sklearn undefined metric warnings if a batch lacks a specific class
warnings.filterwarnings("ignore", category=RuntimeWarning)

from model import Model
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


__author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"


# ==========================================
# ASVspoof 5 IN-MEMORY METRIC ENGINE
# ==========================================

def compute_eer(y_true, y_score):
    """Calculates the Equal Error Rate (EER) mathematically in memory."""
    if len(np.unique(y_true)) < 2:
        return 0.0 # Prevent crash if a batch only contains one class
        
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100 # Return as percentage

def calculate_advanced_metrics(y_true, y_pred):
    """Calculates Accuracy and F1-Score."""
    acc = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0) * 100
    return acc, f1

def compute_minDCF(y_true, y_score, P_spoof=0.05, C_miss=1, C_fa=10):
    """Calculates the minimum Normalized Detection Cost Function (minDCF) for ASVspoof 5 Track 1."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    beta = (C_miss / C_fa) * ((1 - P_spoof) / P_spoof)
    dcf = fnr + beta * fpr
    return np.min(dcf)
# ==========================================

@torch.no_grad()
def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    y_true_all, y_score_all, y_pred_all = [], [], []
    attack_dict = defaultdict(lambda: {'y_true': [], 'y_score': [], 'y_pred': []})

    for batch_x, batch_y, batch_attack in tqdm(dev_loader, desc="Validation Batches", leave=False):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        val_loss += batch_loss.item() * batch_size
        
        # Metric Tracking
        probs = torch.nn.functional.softmax(batch_out, dim=1)[:, 1].detach().cpu().numpy()
        preds = torch.argmax(batch_out, dim=1).detach().cpu().numpy()
        trues = batch_y.cpu().numpy()
        
        y_true_all.extend(trues)
        y_score_all.extend(probs)
        y_pred_all.extend(preds)
        
        # Per-Attack Tracking
        for i, atk in enumerate(batch_attack):
            attack_dict[atk]['y_true'].append(trues[i])
            attack_dict[atk]['y_score'].append(probs[i])
            attack_dict[atk]['y_pred'].append(preds[i])

    val_loss /= num_total
    
    # Global Metrics
    global_eer = compute_eer(y_true_all, y_score_all)
    global_min_dcf = compute_minDCF(y_true_all, y_score_all)
    global_acc, global_f1 = calculate_advanced_metrics(y_true_all, y_pred_all)
    
    # Per-Attack Metrics (Compared against Bonafide)
    per_attack_metrics = {}
    bonafide_scores = np.array(attack_dict['bonafide']['y_score']) if 'bonafide' in attack_dict else np.array([])
    bonafide_trues = np.array(attack_dict['bonafide']['y_true']) if 'bonafide' in attack_dict else np.array([])
    
    for atk, data in attack_dict.items():
        if atk in ['bonafide', 'Unknown', '-']: continue
        
        atk_scores = np.array(data['y_score'])
        atk_trues = np.array(data['y_true'])
        atk_preds = np.array(data['y_pred'])
        
        combined_scores = np.concatenate([bonafide_scores, atk_scores]) if len(bonafide_scores) > 0 else atk_scores
        combined_trues = np.concatenate([bonafide_trues, atk_trues]) if len(bonafide_trues) > 0 else atk_trues
        
        atk_eer = compute_eer(combined_trues, combined_scores)
        atk_acc = accuracy_score(atk_trues, atk_preds) * 100
        per_attack_metrics[atk] = {'EER': atk_eer, 'Acc': atk_acc}

    return val_loss, global_eer, global_min_dcf, global_acc, global_f1, per_attack_metrics


@torch.no_grad()
def produce_evaluation_file(dataset, model, device, save_path, args):
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()

    fname_list = []
    key_list = []
    score_list = []

    for batch_x, utt_id in data_loader:
        fname_list = []
        score_list = []
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)

        batch_out = model(batch_x)

        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

        with open(save_path, "a+") as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write("{} {}\n".format(f, cm))
        fh.close()
    print("Scores saved to {}".format(save_path))

########## Updated the train_epoch function to include AMP ##########
def train_epoch(train_loader, model, lr, optimizer, device, scaler):
    running_loss = 0
    num_total = 0.0
    model.train()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    y_true_all, y_score_all, y_pred_all = [], [], []
    
    for batch_x, batch_y, batch_attack in tqdm(train_loader, desc="Training Batches", leave=False):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            batch_out = model(batch_x)
            batch_loss = criterion(batch_out, batch_y)
            
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += (batch_loss.item() * batch_size)
        
        # Metric Tracking
        probs = torch.nn.functional.softmax(batch_out, dim=1)[:, 1].detach().cpu().numpy()
        preds = torch.argmax(batch_out, dim=1).detach().cpu().numpy()
        y_true_all.extend(batch_y.cpu().numpy())
        y_score_all.extend(probs)
        y_pred_all.extend(preds)
        
    running_loss /= num_total
    eer = compute_eer(y_true_all, y_score_all)
    min_dcf = compute_minDCF(y_true_all, y_score_all)
    acc, f1 = calculate_advanced_metrics(y_true_all, y_pred_all)
    
    return running_loss, eer, min_dcf, acc, f1
########## End of Updated train_epoch function with AMP ##########


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof2021 baseline system")
    # Dataset
    parser.add_argument(
        "--database_path",
        type=str,
        default="/content/databases/LA/",
        help="Change this to user's full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.",
    )
    """
    % database_path/
    %   |- LA
    %      |- ASVspoof2021_LA_eval/flac
    %      |- ASVspoof2019_LA_train/flac
    %      |- ASVspoof2019_LA_dev/flac

 
 
    """

    parser.add_argument(
        "--protocols_path",
        type=str,
        default="content/databases/protocols/ASVspoof_LA_cm_protocols/",
        help="Change with path to user's LA database protocols directory address",
    )
    """
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt
  
    """
    # Model saving and resumption options
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="/content/pretrained_models/xlsr2_300m.pt",
        help="Path to the fairseq wav2vec 2.0 weights",
    )  ########## Added
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Exact path to the .pth checkpoint to resume from",
    )  ######### Added
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./models",
        help="Directory in Google Drive to securely save state dictionaries",
    )  ######### Added
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="./logs",
        help="Directory in Google Drive to securely save TensorBoard telemetry",
    )  ########## Added

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=14)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.000001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--loss", type=str, default="weighted_CCE")
    # model
    parser.add_argument(
        "--seed", type=int, default=1234, help="random seed (default: 1234)"
    )

    parser.add_argument("--model_path", type=str, default=None, help="Model checkpoint")
    parser.add_argument(
        "--comment", type=str, default=None, help="Comment to describe the saved model"
    )
    # Auxiliary arguments
    parser.add_argument(
        "--track", type=str, default="LA", choices=["LA", "PA", "DF"], help="LA/PA/DF"
    )
    parser.add_argument(
        "--eval_output",
        type=str,
        default=None,
        help="Path to save the evaluation result",
    )
    parser.add_argument("--eval", action="store_true", default=False, help="eval mode")
    parser.add_argument(
        "--is_eval", action="store_true", default=False, help="eval database"
    )
    parser.add_argument("--eval_part", type=int, default=0)
    # backend options
    parser.add_argument(
        "--cudnn-deterministic-toggle",
        action="store_false",
        default=True,
        help="use cudnn-deterministic? (default true)",
    )

    parser.add_argument(
        "--cudnn-benchmark-toggle",
        action="store_true",
        default=False,
        help="use cudnn-benchmark? (default false)",
    )

    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument(
        "--algo",
        type=int,
        default=5,
        help="Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]",
    )

    # LnL_convolutive_noise parameters
    parser.add_argument(
        "--nBands",
        type=int,
        default=5,
        help="number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]",
    )
    parser.add_argument(
        "--minF",
        type=int,
        default=20,
        help="minimum centre frequency [Hz] of notch filter.[default=20] ",
    )
    parser.add_argument(
        "--maxF",
        type=int,
        default=8000,
        help="maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]",
    )
    parser.add_argument(
        "--minBW",
        type=int,
        default=100,
        help="minimum width [Hz] of filter.[default=100] ",
    )
    parser.add_argument(
        "--maxBW",
        type=int,
        default=1000,
        help="maximum width [Hz] of filter.[default=1000] ",
    )
    parser.add_argument(
        "--minCoeff",
        type=int,
        default=10,
        help="minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]",
    )
    parser.add_argument(
        "--maxCoeff",
        type=int,
        default=100,
        help="maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]",
    )
    parser.add_argument(
        "--minG",
        type=int,
        default=0,
        help="minimum gain factor of linear component.[default=0]",
    )
    parser.add_argument(
        "--maxG",
        type=int,
        default=0,
        help="maximum gain factor of linear component.[default=0]",
    )
    parser.add_argument(
        "--minBiasLinNonLin",
        type=int,
        default=5,
        help=" minimum gain difference between linear and non-linear components.[default=5]",
    )
    parser.add_argument(
        "--maxBiasLinNonLin",
        type=int,
        default=20,
        help=" maximum gain difference between linear and non-linear components.[default=20]",
    )
    parser.add_argument(
        "--N_f",
        type=int,
        default=5,
        help="order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]",
    )

    # ISD_additive_noise parameters
    parser.add_argument(
        "--P",
        type=int,
        default=10,
        help="Maximum number of uniformly distributed samples in [%].[defaul=10]",
    )
    parser.add_argument(
        "--g_sd", type=int, default=2, help="gain parameters > 0. [default=2]"
    )

    # SSI_additive_noise parameters
    parser.add_argument(
        "--SNRmin",
        type=int,
        default=10,
        help="Minimum SNR value for coloured additive noise.[defaul=10]",
    )
    parser.add_argument(
        "--SNRmax",
        type=int,
        default=40,
        help="Maximum SNR value for coloured additive noise.[defaul=40]",
    )

    ##===================================================Rawboost data augmentation ======================================================================#

    if not os.path.exists("models"):
        os.mkdir("models")
    args = parser.parse_args()

    # make experiment reproducible
    set_random_seed(args.seed, args)

    track = args.track

    assert track in ["LA", "PA", "DF"], "Invalid track given"

    # database
    prefix = "ASVspoof_{}".format(track)
    prefix_2019 = "ASVspoof2019.{}".format(track)
    prefix_2021 = "ASVspoof2021.{}".format(track)

    # define model saving path
    model_tag = "model_{}_{}_{}_{}_{}".format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr
    )
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_save_path = os.path.join("models", model_tag)

    # set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # GPU device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))

    model = Model(args, device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model = model.to(device)
    print("nb_params:", nb_params)

    # set Adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Initialize the AMP GradScaler
    scaler = GradScaler() ########## Added

    ########## Added Initialization and Resumption Block ##########
    # Default starting values for a fresh run
    start_epoch = 0
    best_val_loss = float("inf")

    # Intercept and resume if the argument is passed
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading Full State Checkpoint from '{args.resume}'...")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

            # 1. Restore iterator and metrics
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))

            # 2. Restore computational graph and momentum buffers
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            # Restore AMP Scaling state if it exists
            ########## Added
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
            # 3. Restore deterministic data shuffling variables
            torch.set_rng_state(checkpoint["torch_rng_state"])
            if torch.cuda.is_available() and checkpoint["cuda_rng_state"] is not None:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
            np.random.set_state(checkpoint["np_rng_state"])
            random.setstate(checkpoint["random_rng_state"])

            print(
                f"=> Successfully resumed from checkpoint '{args.resume}' (Fast-forwarding to epoch {start_epoch})"
            )
        else:
            print(
                f"=> WARNING: No checkpoint found at '{args.resume}'. Starting fresh."
            )
    ########## End of Initialization and Resumption Block ##########

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Model loaded : {}".format(args.model_path))

    # evaluation
    if args.eval:
        print("\n[*] Initializing Evaluation Phase on ASVspoof 5 Eval Set...")
        d_label_eval, file_eval = genSpoof_list_ASV5(
            dir_meta=os.path.join(args.protocols_path, 'ASVspoof5.eval.track_1.tsv'), 
            is_train=False, is_eval=True
        )
        print(f"[*] Evaluation utterances loaded: {len(file_eval)}")
        
        eval_set = Dataset_ASVspoof5(
            list_IDs=file_eval, 
            labels_dict=d_label_eval, 
            base_dir=os.path.join(args.database_path, 'flac_E'),
            is_train=False
        )
        eval_loader = DataLoader(
            eval_set, batch_size=args.batch_size, num_workers=8, 
            shuffle=False, drop_last=False, pin_memory=True
        )
        
        eval_loss, eval_eer, eval_min_dcf, eval_acc, eval_f1, per_atk = evaluate_accuracy(eval_loader, model, device)
        
        print("\n" + "="*50)
        print(" ASVspoof 5 FINAL EVALUATION DASHBOARD")
        print("="*50)
        print(f" Global EER      : {eval_eer:.3f}%")
        print(f" Global minDCF   : {eval_min_dcf:.4f}")
        print(f" Global Accuracy : {eval_acc:.2f}%")
        print(f" Global F1-Score : {eval_f1:.2f}%")
        print("-" * 50)
        print(" PER-ATTACK BREAKDOWN (Spoof vs Bonafide):")
        for atk, metrics in sorted(per_atk.items()):
            print(f"   {atk:<5} -> EER: {metrics['EER']:>6.2f}% | Acc: {metrics['Acc']:>6.2f}%")
        print("="*50 + "\n")
        sys.exit(0)

    # define train dataloader
    # ==========================================
    # ASVspoof 5 DATA ROUTING & LOADERS
    # ==========================================
    
    # 1. Training Set Initialization
    d_label_trn, file_train = genSpoof_list_ASV5(
        dir_meta=os.path.join(args.protocols_path, 'ASVspoof5.train.tsv'), 
        is_train=True, is_eval=False
    )
    print(f"[*] Training utterances loaded: {len(file_train)}")
    
    train_set = Dataset_ASVspoof5(
        list_IDs=file_train, 
        labels_dict=d_label_trn, 
        base_dir=os.path.join(args.database_path, 'flac_T'),
        args=args, algo=args.algo, is_train=True
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, num_workers=8, 
        shuffle=True, drop_last=True, pin_memory=True
    )

    # 2. Development Set Initialization
    d_label_dev, file_dev = genSpoof_list_ASV5(
        dir_meta=os.path.join(args.protocols_path, 'ASVspoof5.dev.track_1.tsv'), 
        is_train=False, is_eval=False
    )
    print(f"[*] Development utterances loaded: {len(file_dev)}")
    
    dev_set = Dataset_ASVspoof5(
        list_IDs=file_dev, 
        labels_dict=d_label_dev, 
        base_dir=os.path.join(args.database_path, 'flac_D'),
        is_train=False
    )
    dev_loader = DataLoader(
        dev_set, batch_size=args.batch_size, num_workers=8, 
        shuffle=False, pin_memory=True
    )
    # ==========================================

    ########## Edited the Logging Destination to Be on the Persistent Shared Drive ##########
    # Training and validation
    num_epochs = args.num_epochs

    # Securely route telemetry to the persistent Shared Drive workspace
    secure_log_path = os.path.join(args.logs_dir, model_tag)
    os.makedirs(secure_log_path, exist_ok=True)
    writer = SummaryWriter(secure_log_path)
    ########## End of Edited Logging Destination Block ##########

    for epoch in range(start_epoch, num_epochs):

        # Run Training and Validation
        trn_loss, trn_eer, trn_dcf, trn_acc, trn_f1 = train_epoch(train_loader, model, args.lr, optimizer, device, scaler)
        val_loss, val_eer, val_dcf, val_acc, val_f1, per_atk = evaluate_accuracy(dev_loader, model, device)
        
        # Telemetry Logging
        writer.add_scalar("Loss/Train", trn_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Metrics/Val_EER", val_eer, epoch)
        writer.add_scalar("Metrics/Val_minDCF", val_dcf, epoch)
        
        # Live Terminal Dashboard
        print("\n" + "="*60)
        print(f" EPOCH {epoch} SUMMARY")
        print("="*60)
        print(f" [TRAIN] Loss: {trn_loss:.4f} | EER: {trn_eer:.2f}% | minDCF: {trn_dcf:.4f} | Acc: {trn_acc:.2f}% | F1: {trn_f1:.2f}%")
        print(f" [VALID] Loss: {val_loss:.4f} | EER: {val_eer:.2f}% | minDCF: {val_dcf:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.2f}%")
        print("-" * 60)
        print(" PER-ATTACK VALIDATION EER:")
        
        # Format the per-attack EERs into a clean grid
        atk_strs = [f"{k}: {v['EER']:.2f}%" for k, v in sorted(per_atk.items())]
        for i in range(0, len(atk_strs), 4):
            print("   " + " | ".join(atk_strs[i:i+4]))
        print("="*60)

        # Full State Checkpoint
        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "np_rng_state": np.random.get_state(),
            "random_rng_state": random.getstate(),
        }

        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint_payload, checkpoint_path)
        
        latest_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pth")
        torch.save(checkpoint_payload, latest_path)

        # Best Model Tracker
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_artifact_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_artifact_path)
            print(f"🌟 New Best Model (Loss: {best_val_loss:.4f}) secured at: {best_artifact_path} 🌟\n")
        ########## End of Best Model Tracker Block ##########
        ########## End of Full State Checkpoint Block ##########
