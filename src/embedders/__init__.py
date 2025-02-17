from src.embedders.plm import ESMEmbedder
import re 

esm_embedder_pattern = re.compile('esm_(650m|3b)_(log|cls|gap)')

def get_embedder(feature_type:str):
    ''' Instantiate the appropriate embedder for the feature type.'''
    
    if re.match(esm_embedder_pattern, feature_type) is not None:
        model_size = re.match(esm_embedder_pattern, feature_type).group(1)
        pooler = re.match(esm_embedder_pattern, feature_type).group(2)
        return ESMEmbedder(method=method, model_size=model_size)

    raise Exception(f'get_embedder: The feature type {feature_type} is not recognized.')


