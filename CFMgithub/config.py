exp_configuration = {
    000: {
        'dataset': 'ImageNet',
        'targeted': True,
        'epsilon': 16,
        'alpha': 2,
        'max_iterations': 200,  # "max_iterations"
        'num_images': 500,
        'p': 1.,  # "prob for DI and RE"

        'source_model_names': ['vgg16'],
        'target_model_names': ['inception_v3', 'ResNet50', 'DenseNet121', 'vgg16'],
        'attack_methods': {'CFM-DI-TI-MI': 'CDTM'},
        'visualize': False,
        ####################################
        # Admix
        # 'admix_portion': 0.2,  # 'Admix portion for the mixed image'
        # 'num_mix_samples': 3,  # 'Number of randomly sampled images'
        #####################################

        'mixed_image_type_feature': 'C',  # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature': 'SelfShuffle',  # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature': 'M',  # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature': 0.,  # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature': 0.75,
        'mix_prob': 0.1,
        'divisor': 4,
        # 'divisor': 8,   # The size of tensor a (21) must match the size of tensor b (19) at non-singleton dimension 3

        'channelwise': True,
        'mixup_layer': 'conv_linear_include_last',
        #####################################
        'comment': 'CFM-RDI Main Result'
    },
    1: {    # everywhere
        'dataset': 'ImageNet',
        'targeted': True,
        'epsilon': 16,
        'alpha': 2,
        'max_iterations': 280,  # "max_iterations"
        'num_images': 500,
        'p': 1.,  # "prob for DI and RE"

        'source_model_names': ['inception_v3'],
        'target_model_names': ['inception_v3', 'ResNet50', 'DenseNet121', 'vgg16'],
        'attack_methods': {'CFM-DI-TI-MI': 'CDTM'},
        'visualize': False,

        'mixed_image_type_feature': 'C',  # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature': 'SelfShuffle',  # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature': 'M',  # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature': 0.,  # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature': 0.75,
        'mix_prob': 0.1,
        'divisor': 4,
        # 'divisor': 8,   # The size of tensor a (21) must match the size of tensor b (19) at non-singleton dimension 3

        'channelwise': True,
        'mixup_layer': 'conv_linear_include_last',
        #####################################
        'comment': 'CFM-RDI Main Result'
    },
    2: {    # everywhere
        'dataset': 'ImageNet',
        'targeted': True,
        'epsilon': 16,
        'alpha': 2,
        'max_iterations': 200,  # "max_iterations"
        'num_images': 200,
        'p': 1.,  # "prob for DI and RE"

        'source_model_names': ['ResNet50'],
        'target_model_names': ['inception_v3', 'ResNet50', 'DenseNet121', 'vgg16'],
        'attack_methods': {'CFM-DI-TI-MI': 'CDTM'},
        'visualize': False,

        'mixed_image_type_feature': 'C',  # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature': 'SelfShuffle',  # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature': 'M',  # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature': 0.,  # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature': 0.75,
        'mix_prob': 0.1,
        'divisor': 4,
        # 'divisor': 8,   # error: The size of tensor a (21) must match the size of tensor b (19) at non-singleton dimension 3

        'channelwise': True,
        'mixup_layer': 'conv_linear_include_last',
        #####################################
        'comment': 'CFM-RDI Main Result'
    },
    578: {
        'dataset': 'ImageNet',
        'targeted': True,
        'epsilon': 16,
        'alpha': 2,
        'max_iterations': 200,  # "max_iterations"
        'num_images': 500,
        'p': 1.,  # "prob for DI and RE"

        'source_model_names': ['ResNet50'],
        'target_model_names': ['ResNet18', 'ResNet50', 'vgg16', 'inception_v3', 'DenseNet121'],
        'attack_methods': {'CFM-RDI-TI-MI': 'CRTM'},
        'number_of_v_samples': 5,
        'number_of_si_scales': 5,
        'visualize': False,
        ####################################
        # Admix
        'admix_portion': 0.2,  # 'Admix portion for the mixed image'
        'num_mix_samples': 3,  # 'Number of randomly sampled images'
        #####################################

        'mixed_image_type_feature': 'C',  # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature': 'SelfShuffle',  # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature': 'M',  # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature': 0.,  # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature': 0.75,
        'mix_prob': 0.1,
        'divisor': 4,

        'channelwise': True,
        'mixup_layer': 'conv_linear_include_last',
        #####################################
        'comment': 'CFM-RDI Main Result'
    }
}