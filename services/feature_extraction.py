import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Global model instance (loaded once)
_model = None
_transform = None


def _load_model(embedding_dim: int = 128):
    """Load pretrained ResNet18 and modify for embedding extraction."""
    global _model, _transform

    if _model is not None:
        return _model, _transform

    logger.info("Loading ResNet18 model...")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Replace final FC layer with embedding layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, embedding_dim),
        nn.BatchNorm1d(embedding_dim),
    )

    model.eval()
    _model = model

    _transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    logger.info(f"Model loaded. Embedding dim: {embedding_dim}")
    return _model, _transform


def extract_embedding(roi_image: np.ndarray, embedding_dim: int = 128) -> dict:
    """
    Extract feature embedding from ROI image using ResNet18.

    Args:
        roi_image: BGR image (3-channel, any size)
        embedding_dim: Dimension of output embedding

    Returns:
        dict with 'embedding' (list of floats) and 'success'
    """
    model, transform = _load_model(embedding_dim)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)

    # Apply transforms
    tensor = transform(rgb).unsqueeze(0)  # Add batch dimension

    # Extract embedding
    with torch.no_grad():
        embedding = model(tensor)

    # L2 normalize the embedding
    embedding = nn.functional.normalize(embedding, p=2, dim=1)

    embedding_list = embedding.squeeze().tolist()
    logger.info(f"Embedding extracted: {len(embedding_list)}D")

    return {
        "success": True,
        "embedding": embedding_list,
    }
