import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm

from config import parse_args
from models.hybridnet import HybridNetwork
from datasets.keypoint_dataset import HybridKeypointDataset
from datasets.imagenet_dataset import ImageNetKeypointDataset, imagenet_collate_fn
from utils.train_utils import hybrid_collate, evaluate

def main():
    args = parse_args()

    # ----------------------- Initialization -----------------------
    # Distributed training configuration
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # ----------------------- Data Preparation -----------------------
    try:
        # Select appropriate dataset class and batch processing function based on dataset type
        if args.dataset_type == 'imagenet':
            print(f"Using ImageNet dataset for training")
            train_dataset = ImageNetKeypointDataset(
                data_root=args.data_dir,
                split='train',
                img_size=args.img_size,
                augment=True,
                args=args
            )
            
            val_dataset = ImageNetKeypointDataset(
                data_root=args.val_dir if args.val_dir != 'val' else args.data_dir,
                split='val',
                img_size=args.img_size,
                augment=False,
                args=args
            )
            
            collate_fn = imagenet_collate_fn
        else:
            print(f"Using custom keypoint dataset for training")
            train_dataset = HybridKeypointDataset(
                data_root=args.data_dir,
                img_size=args.img_size,
                augment=True,
                args=args
            )
            
            val_dataset = HybridKeypointDataset(
                data_root=args.val_dir,
                img_size=args.img_size,
                augment=False,
                args=args
            )
            
            collate_fn = hybrid_collate
        
        # Test dataset
        if args.dataset_type == 'imagenet':
            test_dataset = ImageNetKeypointDataset(
                data_root=args.test_dir if args.test_dir != 'test' else args.data_dir,
                split='val',
                img_size=args.img_size,
                augment=False,
                args=args
            )
        else:
            test_dataset = HybridKeypointDataset(
                data_root=args.test_dir,
                img_size=args.img_size,
                augment=False,
                args=args
            )
            
    except Exception as e:
        print(f"Dataset initialization failed: {str(e)}")
        sys.exit(1)

    # ----------------------- Distributed Loading -----------------------
    train_sampler = dist.DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    val_sampler = dist.DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # ----------------------- Model Initialization -----------------------
    model = HybridNetwork(args).to(local_rank)
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = optim.AdamW(model.parameters(), lr=args.lr_keynet * world_size)
    scaler = torch.cuda.amp.GradScaler()

    # ----------------------- Resume Training -----------------------
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        try:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
            checkpoint = torch.load(args.resume, map_location=map_location)

            # Compatibility loading
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', optimizer.state_dict()))
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            args.test_dir = checkpoint.get('test_dir', args.test_dir)

            if rank == 0:
                print(f" Successfully loaded checkpoint: {args.resume}")
                print(f"   - Starting epoch: {start_epoch}")
                print(f"   - Best validation loss: {best_val_loss:.4f}")

        except Exception as e:
            print(f"❌ Failed to load checkpoint: {str(e)}")
            print("⚠️ Starting training with initial parameters")

    # ----------------------- Training Loop -----------------------
    early_stop_counter = 0
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0

        # ----------------------- Training Phase -----------------------
        with tqdm(train_loader, desc=f'Train Epoch {epoch + 1}/{args.epochs}',
                 bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                 disable=(rank != 0)) as pbar:

            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(local_rank, non_blocking=True)
                targets = targets.to(local_rank, non_blocking=True)

                # Mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = F.mse_loss(outputs, targets)

                # Backpropagation
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Update progress
                train_loss += loss.item()
                avg_loss = train_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'batch_loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })

        # ----------------------- Validation Phase -----------------------
        val_loss = evaluate(model, val_loader, local_rank,
                           desc=f'Validation [{epoch + 1}/{args.epochs}]')

        # Distributed synchronization
        if world_size > 1:
            train_loss = torch.tensor(train_loss).cuda()
            val_loss_tensor = torch.tensor(val_loss).cuda()
            dist.all_reduce(train_loss)
            dist.all_reduce(val_loss_tensor)
            train_loss = train_loss.item() / world_size
            val_loss = val_loss_tensor.item() / world_size

        if rank == 0:
            # ----------------------- Model Saving -----------------------
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'test_dir': args.test_dir,
                'config': vars(args)
            }

            # Periodic saving
            torch.save(checkpoint, os.path.join(args.save_dir, f'epoch_{epoch}.pth'))

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
                print(f"New best model @ Epoch {epoch + 1}: Validation loss {val_loss:.4f}")

            # Early stopping mechanism
            else:
                early_stop_counter += 1
                if early_stop_counter >= 3:
                    print(f"Early stopping triggered @ Epoch {epoch + 1}")
                    break

    # ----------------------- Final Test -----------------------
    if rank == 0:
        print("\n🚀 Final test phase...")
        best_model_path = os.path.join(args.save_dir, 'best_model.pth')
        final_model_path = os.path.join(args.save_dir, f'epoch_{args.epochs - 1}.pth')

        # Prefer best model
        if os.path.exists(best_model_path):
            test_model_path = best_model_path
        # Secondary option: final epoch model
        elif os.path.exists(final_model_path):
            test_model_path = final_model_path
            print("⚠️ Best model not found, using final epoch model for testing")
        # Use restored model
        elif args.resume:
            test_model_path = args.resume
            print("⚠️ Testing with restored checkpoint")
        else:
            raise FileNotFoundError("No model found for testing")

        try:
            checkpoint = torch.load(test_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            test_loss = evaluate(model, test_loader, local_rank, desc="Final test")
            print(f"🏁 Test completed | Test loss: {test_loss:.4f}")
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    main() 