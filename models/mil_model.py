import torch
import torch.nn as nn


class MB_sMIL(nn.Module):
    """
    Multi-branch Multiple Instance Learning model with self-attention mechanism.

    This model uses instance-level attention followed by class-specific attention
    branches to aggregate patch features into slide-level predictions.
    """

    def __init__(
        self, feature_dim, num_classes, hidden_dim=256, dropout=0.2, num_heads=16
    ):
        """
        Args:
            feature_dim: Dimension of input features
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for attention branches
            dropout: Dropout rate
            num_heads: Number of attention heads
        """
        super().__init__()

        self.num_classes = num_classes
        self.instance_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True
        )

        # Create separate attention branches for each class
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid(),
                )
                for _ in range(num_classes)
            ]
        )

        # Final classifier
        self.classifiers = nn.Sequential(
            nn.Linear(feature_dim * num_classes, hidden_dim),
            nn.GELU(),
            nn.GroupNorm(8, hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        """
        Forward pass.

        Args:
            features: Input features [N, feature_dim] where N is number of patches

        Returns:
            logits: Classification logits [1, num_classes]
            attention_all: Attention weights [N, num_classes]
        """
        # Instance-level attention
        attended_features, _ = self.instance_attention(features, features, features)
        attended_features = attended_features + features  # Residual connection
        attended_features = self.dropout(attended_features)

        pooled_all = []
        attention_all = []

        # Class-specific attention branches
        for branch in self.branches:
            attention_weights = branch(attended_features)
            gated_features = attended_features * attention_weights
            pooled_features = gated_features.mean(dim=0).unsqueeze(0)
            pooled_features = self.dropout(pooled_features)
            attention_all.append(attention_weights)
            pooled_all.append(pooled_features)

        # Concatenate pooled features from all branches
        pooled_all = torch.cat(pooled_all, dim=1)
        logits = self.classifiers(pooled_all)
        attention_all = torch.cat(attention_all, dim=1)

        return logits, attention_all


def create_mil_model(config):
    """
    Create MIL model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        MIL model instance
    """
    mil_config = config["model"]["mil"]
    num_classes = config["data"]["num_classes"]

    model = MB_sMIL(
        feature_dim=mil_config["feature_dim"],
        num_classes=num_classes,
        hidden_dim=mil_config["hidden_dim"],
        dropout=mil_config["dropout"],
        num_heads=mil_config["num_attention_heads"],
    )

    return model
