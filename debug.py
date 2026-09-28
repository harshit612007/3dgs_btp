import traceback, sys
import render_video, train
import argparse

args = argparse.Namespace(dataset_path='datasets/lego', ply_path='outputs/lego_128_10k_max/optimized_scene.ply', output_dir='outputs/lego_128_10k_max')

try:
    render_video.main(args)
except Exception as e:
    tb = e.__traceback__
    while tb.tb_next: tb = tb.tb_next
    f = tb.tb_frame
    def s(name):
        obj = f.f_locals.get(name)
        if obj is not None and hasattr(obj, 'shape'):
            return str(obj.shape)
        return "N/A"
    
    print('out_color:', s('out_color'))
    print('transmittance:', s('transmittance'))
    print('weight:', s('weight'))
    print('c_chunk:', s('c_chunk'))
    print('alpha:', s('alpha'))
    print('dx:', s('dx'))
    print('dist2:', s('dist2'))
    print('grid:', s('grid'))
