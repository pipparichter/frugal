import re 

esm_embedder_pattern = re.compile('esm_(650m|3b)_(log|cls|gap)')
pt5_embedder_pattern = re.compile('pt5_3b_gap')

def get_embedder(feature_type:str):
    ''' Instantiate the appropriate embedder for the feature type.'''
    
    if re.match(esm_embedder_pattern, feature_type) is not None:
        model_size = re.match(esm_embedder_pattern, feature_type).group(1)
        pooler = re.match(esm_embedder_pattern, feature_type).group(2)
        return ESMEmbedder(pooler=pooler, model_size=model_size)

    if re.match(pt5_embedder_pattern, feature_type) is not None:
        return ProtT5Embedder()

    raise Exception(f'get_embedder: The feature type {feature_type} is not recognized.')


