import torch
import torch.nn.functional as F
from tqdm import tqdm

def hybrid_collate(batch):
    """Safe batch processing function"""
    inputs = torch.stack([item[0] for item in batch]) # [B,4,512,512]
    targets = torch.stack([item[1] for item in batch]) # [B,1,512,512]
    return inputs, targets

def evaluate(model, loader, device, desc="Evaluating"):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        with tqdm(loader, desc=desc,
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as val_progress:
            for batch_idx, (inputs, targets) in enumerate(val_progress):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = F.mse_loss(outputs, targets)
                total_loss += loss.item()

                # Real-time validation loss display
                avg_loss = total_loss / (batch_idx + 1)
                val_progress.set_postfix({'val_loss': f'{avg_loss:.4f}'})

    return total_loss / len(loader.dataset) 