#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
import argparse
import os
import sys

import onnx_graphsurgeon as gs
import numpy as np
import onnx


def modify(input: str, output: str, downsample_ratio: float = 0.25) -> None:
    print(f'\nonnx load: {input}')
    graph = gs.import_onnx(onnx.load(input))

    _print_graph(graph)

    # update node Resize_3: scales
    resize_3 = [n for n in graph.nodes if n.name == 'Resize_3'][0]
    print()
    print(resize_3)

    scales = gs.Constant('388',
        np.asarray([1, 1, downsample_ratio, downsample_ratio], dtype=np.float32))

    resize_3.inputs = [i if i.name != '388' else scales for i in resize_3.inputs]
    print()
    print(resize_3)

    # remove input downsample_ratio
    graph.inputs = [i for i in graph.inputs if i.name != 'downsample_ratio']

    # remove node Concat_2
    concat_2 = [n for n in graph.nodes if n.name == 'Concat_2'][0]
    concat_2.outputs.clear()

    # remove unused nodes/tensors
    graph.cleanup()

    onnx.save(gs.export_onnx(graph), output)


def _print_graph(graph: gs.Graph) -> None:
    print(f'\ngraph.opset={graph.opset}')
    print('\ngraph.inputs')
    for i in graph.inputs:
        print(f'  {i}')
    print('\ngraph.outputs')
    for o in graph.outputs:
        print(f'  {o}')


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('--downsample-ratio', type=float, default=0.25)
    parser.add_argument('--input-size', type=int, default=None, nargs=2,
        help='auto downsample ratio by input size')

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f'input not found: {args.input}')
    if args.output is None:
        root, ext = os.path.splitext(args.input)
        args.output = f'{root}_modified{ext}'
    if args.input_size is not None:
        w, h = args.input_size
        args.downsample_ratio = min(512 / max(h, w), 1)

    print('Args')
    print(f'  input: {args.input}')
    print(f'  output: {args.output}')
    print(f'  downsample_ratio: {args.downsample_ratio}')
    print(f'  input_size: {args.input_size}')
    return args


def _main():
    args = _parse_args()
    modify(args.input, args.output, args.downsample_ratio)


if __name__ == '__main__':
    _main()


# rvm_mobilenetv3_fp32.onnx
#  https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx
#
# python -m onnxsim rvm_mobilenetv3_fp32.onnx rvm_mobilenetv3_fp32_sim.onnx \
# --input-shape src:1,3,1080,1920 r1i:1,1,1,1 r2i:1,1,1,1 r3i:1,1,1,1 r4i:1,1,1,1
#
# python rvm_onnx_modify.py -i rvm_mobilenetv3_fp32_sim.onnx --input-size 1920 1280
#
# trtexec --onnx=rvm_mobilenetv3_fp32_sim_modified.onnx --fp16 --workspace=64 --saveEngine=rvm_mobilenetv3_fp32_sim_modified.engine
