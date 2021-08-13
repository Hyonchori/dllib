
import torch
import copy


def model_info(model, verbose=False, input_shape=(3, 640, 640), batch_size=32):
    n_p = sum(x.numel() for x in model.parameters())  # number of parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number of gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    from thop import profile
    print(model.mode)
    if model.mode in ["backbone", "detector"]:
        img = torch.zeros((1, *input_shape), device=next(model.parameters()).device)
        size = (batch_size, *img.shape[1:])

    elif model.mode in ["neck", "head"]:
        img = [torch.zeros((1, *shape), device=next(model.parameters()).device) for shape in input_shape]
        size = [(batch_size, *im.shape[1:]) for im in img]
    flops = profile(copy.deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2
    fs = ", {:.1f} GFLOPs given size{}".format(flops * batch_size, size)

    print(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}\n")