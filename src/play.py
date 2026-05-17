import requests
import pathlib
from rich import print
from io import BytesIO
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

def build_model(backbone_name: str) -> nn.Module:
    model = getattr(models, backbone_name)(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(in_features, 1), nn.Sigmoid()
    )

    return model

def load_model(checkpoint_path: str, backbone: str) -> nn.Module:
    model = build_model(backbone)
    ckpt  = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(ckpt["model_state"])
    model.to("cuda").eval()
    val_loss = ckpt.get('val_loss')
    val_str  = f"{val_loss:.4f}" if val_loss is not None else "N/A"
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} (val_loss={val_str})")
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize(int(256 * 1.15)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def score_image(model, transform, path: str) -> float:
    img   = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to("cuda")
    score  = model(tensor).item()
    return score

if __name__ == "__main__":
    model = load_model(str(pathlib.Path(__file__).parent.parent / "checkpoints/best_model.pt"), "convnext_base")
    tf = get_transform()
    
    temp_dir = pathlib.Path(__file__).parent.parent / "temp"

    r = requests.get("https://scrandle.com/practice")
    data = r.json()
    points = 0

    for i, curr_round in enumerate(data):
        print(f"[bold green]Beginning round #{i+1}...[/bold green]")
        print(f"[blue]Scran #1:\nTitle: {curr_round[0]['title']}\nRating: {curr_round[0]['rating']}\nURL: {curr_round[0]['images_new'][0]}[/blue]\n")
        print(f"[blue]Scran #2:\nTitle: {curr_round[1]['title']}\nRating: {curr_round[1]['rating']}\nURL: {curr_round[1]['images_new'][0]}[/blue]\n")

        scran_img_1 = temp_dir / "img0.jpeg"
        scran_img_2 = temp_dir / "img1.jpeg"

        scran_data = requests.get(curr_round[0]["images_new"][0])
        with Image.open(BytesIO(scran_data.content)) as im:
            im.save(scran_img_1, "JPEG")

        scran_data = requests.get(curr_round[1]["images_new"][0])
        with Image.open(BytesIO(scran_data.content)) as im:
            im.save(scran_img_2, "JPEG")

        s1 = score_image(model, tf, str(scran_img_1))
        s2 = score_image(model, tf, str(scran_img_2))

        if s1 > s2:
            print(f"[bold yellow]Predicted:\nScran #1 ({s1:.3f}) beats Scran #2 ({s2:.3f})[/bold yellow]")
            if curr_round[0]['rating'] > curr_round[1]['rating']:
                points+=1
                print("[bold green]Correct guess![/bold green]")
            else:
                print("[bold red]Incorrect guess![/bold red]")

        else:
            print(f"[bold yellow]Predicted:\nScran #2 ({s2:.3f}) beats Scran #1 ({s1:.3f})[/bold yellow]")
            if curr_round[0]['rating'] < curr_round[1]['rating']:
                points+=1
                print("[bold green]Correct guess![/bold green]")
            else:
                print("[bold red]Incorrect guess![/bold red]")

        print()
        scran_img_1.unlink()
        scran_img_2.unlink()

    print(f"[bold green]Final Total: {points}/10 points[/bold green]")