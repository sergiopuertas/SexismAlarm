import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import StepLR
from ..model import LSTMModel


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    destroy_process_group()


def train_model(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config

        world_size = torch.cuda.device_count()
        rank = int(os.environ.get("LOCAL_RANK", 0))

        setup(rank, world_size)

        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # Load data and model
        path = "model/train/"
        train_dataset = torch.load(f"{path}train_dataset.pt")
        val_dataset = torch.load(f"{path}val_dataset.pt")
        pretrained_embeddings = torch.load(f"{path}embeddings.pt")

        model = LSTMModel(config.emb_dim, pretrained_embeddings, config.dropout)
        model = model.to(device)
        model = DDP(model, device_ids=[rank])

        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, sampler=train_sampler
        )
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = StepLR(optimizer, step_size=2, gamma=config.lr_decay)

        best_val_loss = float("inf")
        for epoch in range(config.epochs):
            model.train()
            epoch_loss = 0
            for texts, labels in tqdm(train_loader):
                texts, labels = texts.to(device), labels.float().to(device)

                outputs = model(texts)
                loss = nn.BCELoss()(outputs.squeeze(), labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            val_loss = validate_model(model, val_loader, device)

            if rank == 0:
                wandb.log(
                    {"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss}
                )

                # Guardar el modelo si es el mejor hasta ahora
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_path = (
                        f"{path}model_checkpoints/model_{run.id}_epoch_{epoch}.pt"
                    )
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.module.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": val_loss,
                        },
                        model_path,
                    )
                    wandb.save(model_path)  # Esto subirá el archivo a wandb

            scheduler.step()

        cleanup()


def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.float().to(device)
            outputs = model(texts)
            loss = nn.BCELoss()(outputs.squeeze(), labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)


def main():
    with open("..key.txt", "r", encoding="UTF-8") as f:
        wandb_key = f.read().strip()
    wandb.login(key=wandb_key)

    # Asegúrate de que el directorio para los checkpoints existe
    os.makedirs("model/train/model_checkpoints", exist_ok=True)

    sweep_config = {
        "method": "random",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"min": 1e-4, "max": 1e-3},
            "dropout": {"min": 0.2, "max": 0.5},
            "weight_decay": {"min": 1e-6, "max": 1e-4},
            "lr_decay": {"min": 0.1, "max": 0.5},
            "epochs": {"value": 5},
            "batch_size": {"values": [32, 64, 128]},
            "emb_dim": {"values": [100, 200, 300]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="sexism_detector_lstm_distributed")
    wandb.agent(sweep_id, train_model, count=5)


if __name__ == "__main__":
    main()
