import ml_collections

def get_MCFT_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 480  # KV_size = Q1 + Q2 + Q3 + Q4
    # config.KV_size = [32, 64, 128, 256]  # KV_size = Q1, Q2, Q3, Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embedding_channels = 32 * config.transformer.num_heads
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 32 # base channel of U-Net
    config.n_classes = 1
    return config
