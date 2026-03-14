import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score, classification_report


def quadratic_kappa(y_true, y_pred):
    """Quadratic weighted kappa — official APTOS 2019 metric."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def evaluate(model, loader, criterion, device):
    """
    Run model on loader, return:
        avg_loss, accuracy, kappa, all_preds, all_labels
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    kappa    = quadratic_kappa(all_labels, all_preds)

    return avg_loss, accuracy, kappa, all_preds, all_labels


def print_classification_report(all_labels, all_preds, class_names):
    print(classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4
    ))
