
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
    img = torch.zeros((1, *input_shape), device=next(model.parameters()).device)
    flops = profile(copy.deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2
    size = img.shape
    fs = ", {:.1f} GFLOPs given size{}".format(flops * batch_size, (batch_size, *size[1:]))

    print(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}\n")