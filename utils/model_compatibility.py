# utils/model_compatibility.py

def check_model_compatibility(model_path):
    if not os.path.isdir(model_path):
        print(f"{model_path} is not a directory")
        return None
    
    info = {
        'path': model_path,
        'has_config': os.path.exists(os.path.join(model_path, 'config.json')),
        'has_weights': os.path.exists(os.path.join(model_path, 'pytorch_model.bin')),
        'has_model_info': os.path.exists(os.path.join(model_path, 'model_info.json')),
        'has_vocabularies': os.path.exists(os.path.join(model_path, 'vocabularies')),
        'has_processor': os.path.exists(os.path.join(model_path, 'preprocessor_config.json'))
    }

    model_info_path = os.path.join(model_path, 'model_info.json')
    if info['has_model_info']:
        try:
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            info.update({
                'hierarchical_mode': model_info.get('hierarchical_mode', 'unknown'),
                'model_description': model_info.get('model_description', 'unknown'),
                'components': model_info.get('components_present', {}),
                'enhancements': model_info.get('enhancement_modules', {})
            })
        except Exception as e:
            print(f"Could not load model info: {e}")
    config_path = os.path.join(model_path, 'config.json')
    if info['has_config']:
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            info['config_hierarchical_mode'] = config.get('hierarchical_mode', 'unknown')
            info['vocab_sizes'] = {
                'base': config.get('base_char_vocab_size', 0),
                'diacritic': config.get('diacritic_vocab_size', 0),
                'combined': config.get('combined_char_vocab_size', 0)
            }
        except Exception as e:
            print(f"Could not load config: {e}")
    
    return info

def print_model_compatibility_report(model_path):
    info = check_model_compatibility(model_path)
    
    if not info:
        return

    if 'hierarchical_mode' in info:
        print(f"Mode: {info['hierarchical_mode']}")
        
        if 'enhancements' in info:
            for name, enabled in info['enhancements'].items():
                status = "ok" if enabled else "no"
                print(f"{status} {name}")
            print()
        
        if 'vocab_sizes' in info:
            for vocab_type, size in info['vocab_sizes'].items():
                print(f"{vocab_type}: {size}")
            print()
    
    if info['has_config'] and info['has_weights']:
        print(" Can load with HierarchicalCtcMultiScaleOcrModel.from_pretrained()")
    if info['has_model_info']:
        print(" Full model info available")
    if info['has_vocabularies']:
        print(" Separate vocabulary files available")
    if not info['has_processor']:
        print("No processor config - may need to specify manually")
