def create_model(opt):
    from .model import only_seg_V1

    # if opt.model == 'local3D':
    #     from .PTN_model3D import PTN
    #     model = PTN()
    model = only_seg_V1(opt)

    return model
