import argparse
import logging
from pprint import pprint

import torch
import torch_pruning as tp

from test_demo import _select_model, run

logger = logging.getLogger("NTIRE2022-EfficientSR")
prune_ignores = {
    0: lambda m: [m.upsampler[0]],
    -3: lambda m: [m.upsampler[0]],
    -4: lambda m: [m.upsampler[0]],
    -5: lambda m: [m.upsampler[0]],
    -6: lambda m: [m.upsampler[0]],
    -7: lambda m: [m.upsampler[0], m.up_conv1[0], m.up_conv2[0], m.up_conv3[0]],
    -8: lambda m: [m.upsampler[0]],
    -9: lambda m: [m.upsampler[0]],
    -10: lambda m: [m.upsampler[0]],
    -11: lambda m: [m.upsampler[0]],
}


def main(args):
    # --------------------------------
    # load model
    # --------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name, data_range, tile = _select_model(args, device)
    model = model.to(device)

    # --------------------------------
    # load example inputs
    # --------------------------------
    input_dim = (3, 256, 256)
    example_inputs = torch.FloatTensor(1, *input_dim).to(device)
    ignored_layers = prune_ignores[args.model_id](model) if args.model_id in prune_ignores else []

    '''
    from torchvision.models import resnet18 as entry
    model = entry(pretrained=True)
    print(model)
    # Global metrics
    example_inputs = torch.randn(1, 3, 224, 224)
    ignored_layers = [model.fc]
    '''

    # --------------------------------
    # load pruner
    # --------------------------------
    iterative_steps = 16
    imp = tp.importance.MagnitudeImportance(p=2)
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,  # progressive pruning
        ch_sparsity=0.5,  # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
    )

    # --------------------------------
    # pruning
    # --------------------------------
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    for i in range(iterative_steps):
        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print(model(example_inputs).shape)
        print(
            "  Iter %d/%d, Params: %.2f M => %.2f M"
            % (i + 1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
        )
        print(
            "  Iter %d/%d, MACs: %.2f G => %.2f G"
            % (i + 1, iterative_steps, base_macs / 1e9, macs / 1e9)
        )
        # finetune your model here
        # finetune(model)
        # ...
        # Test in valid set
        with torch.no_grad():
            results = run(model, model_name, data_range, tile, logger, device, args, mode="valid")
            print(
                'valid_memory', results['valid_memory'],
                'valid_ave_runtime', results['valid_ave_runtime'],
                'valid_ave_psnr', results['valid_ave_psnr']
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NTIRE2022-EfficientSR")
    parser.add_argument("--lr_dir", default="/cluster/work/cvl/yawli/data/NTIRE2022_Challenge", type=str)
    parser.add_argument("--hr_dir", default="/cluster/work/cvl/yawli/data/NTIRE2022_Challenge", type=str)
    parser.add_argument("--save_dir", default="/cluster/work/cvl/yawli/data/NTIRE2022_Challenge/results", type=str)
    parser.add_argument("--model_id", default=0, type=int)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--include_test", action="store_true", help="Inference on the DIV2K test set")
    parser.add_argument("--upscale", default=4, type=int)
    parser.add_argument("--ssim", action="store_true", help="Calculate SSIM")
    parser.add_argument("--onnx", default=None, type=str, help="Save to onnx")

    args = parser.parse_args()
    pprint(args)

    main(args)
