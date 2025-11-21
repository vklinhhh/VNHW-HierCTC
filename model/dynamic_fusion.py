# model/dynamic_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)

class DynamicMultiScaleFusion(nn.Module):
    def __init__(self, encoder_output_size, num_layers=2, fusion_dim=None, use_gate=True):
        super().__init__()
        self.encoder_output_size = encoder_output_size
        self.num_layers = num_layers
        self.fusion_dim = fusion_dim if fusion_dim is not None else encoder_output_size
        self.use_gate = use_gate
        
        self.context_encoder = nn.Sequential(
            nn.Linear(encoder_output_size * num_layers, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_layers),
        )

        if self.use_gate:
            self.gate_generators = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(encoder_output_size, encoder_output_size // 2),
                    nn.LayerNorm(encoder_output_size // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(encoder_output_size // 2, 1),
                )
                for _ in range(num_layers)
            ])
        
        self.fusion_projection = nn.Sequential(
            nn.Linear(encoder_output_size, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.Dropout(0.1)
        )
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features_list):
        batch_size, seq_length, _ = features_list[0].shape
        global_features = []
        for features in features_list:
            mean_features = torch.mean(features, dim=1)  # [batch_size, encoder_output_size]
            global_features.append(mean_features)
        
        # [batch_size, encoder_output_size * num_layers]
        concatenated_global = torch.cat(global_features, dim=-1)
        
        # [batch_size, num_layers]
        global_weights = self.context_encoder(concatenated_global)
        global_weights = F.softmax(global_weights, dim=-1)
        
        globally_weighted = []
        for i, features in enumerate(features_list):
            # [batch_size, 1, 1] * [batch_size, seq_length, encoder_output_size]
            layer_weight = global_weights[:, i:i+1, None]
            weighted_features = features * layer_weight.expand_as(features)
            globally_weighted.append(weighted_features)

        if self.use_gate:
            gated_features = []
            for i, features in enumerate(features_list):
                # [batch_size, seq_length, 1]
                gates = torch.sigmoid(self.gate_generators[i](features))
                # [batch_size, seq_length, encoder_output_size]
                gated = features * gates
                gated_features.append(gated)
            
            fused_features = sum(gated_features)
        else:

            fused_features = sum(globally_weighted)
        
        fused_features = self.fusion_projection(fused_features)
        
        return fused_features
    
    def extra_repr(self):
        return f"encoder_size={self.encoder_output_size}, num_layers={self.num_layers}, fusion_dim={self.fusion_dim}, use_gate={self.use_gate}"


class LocalFeatureEnhancer(nn.Module):
    def __init__(self, feature_dim, num_diacritics=None, use_spatial_attention=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_spatial_attention = use_spatial_attention
        self.num_diacritics = num_diacritics
        
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        self.num_heads = 4
        self.head_dim = feature_dim // self.num_heads
        assert self.head_dim * self.num_heads == feature_dim, "feature_dim must be divisible by num_heads"

        self.pattern_query = nn.Linear(feature_dim, feature_dim)
        self.pattern_key = nn.Linear(feature_dim, feature_dim)
        self.pattern_value = nn.Linear(feature_dim, feature_dim)
        self.pattern_output = nn.Linear(feature_dim, feature_dim)

        if use_spatial_attention:
            self.region_attention = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.LayerNorm(feature_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim // 2, 3)  # 3 regions: above, middle, below
            )
            
            self.region_enhancers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.LayerNorm(feature_dim),
                    nn.GELU(),
                    nn.Dropout(0.1)
                )
                for _ in range(3)
            ])
        
        if num_diacritics and num_diacritics > 0:
            self.diacritic_prototypes = nn.Parameter(
                torch.randn(num_diacritics, feature_dim)
            )
            
            self.diacritic_detector = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim, num_diacritics)
            )
            nn.init.xavier_uniform_(self.diacritic_prototypes)

        self.final_norm = nn.LayerNorm(feature_dim)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features):
        batch_size, seq_length, _ = features.shape
        transformed = self.feature_transform(features)
        
        # multi-head attention for fine-grained pattern detection
        # Compute query, key, value
        query = self.pattern_query(features).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.pattern_key(features).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.pattern_value(features).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        pattern_context = torch.matmul(attention_weights, value)
        pattern_context = pattern_context.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        pattern_output = self.pattern_output(pattern_context)

        enhanced = transformed + pattern_output
        
        # spatial attention
        if self.use_spatial_attention:
            region_weights = F.softmax(self.region_attention(enhanced), dim=-1)  # [batch, seq_len, 3]
            region_enhanced = torch.zeros_like(enhanced)
            for i, enhancer in enumerate(self.region_enhancers):
                region_output = enhancer(enhanced)
                region_enhanced += region_weights[:, :, i:i+1] * region_output
            
            enhanced = enhanced + region_enhanced
        
        # diacritic-specific enhancement
        if hasattr(self, 'diacritic_prototypes') and hasattr(self, 'diacritic_detector'):
            diacritic_logits = self.diacritic_detector(enhanced)  # [batch, seq_len, num_diacritics]
            diacritic_probs = F.softmax(diacritic_logits, dim=-1)

            prototype_features = torch.matmul(diacritic_probs, self.diacritic_prototypes)  # [batch, seq_len, feature_dim]
            enhanced = enhanced + 0.2 * prototype_features
        
        enhanced = self.final_norm(enhanced)
        
        return enhanced
    
    def extra_repr(self):
        return f"feature_dim={self.feature_dim}, num_heads={self.num_heads}, use_spatial_attention={self.use_spatial_attention}"