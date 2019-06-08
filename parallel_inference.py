import multiprocessing
from utils import ProgressBar
import utils
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import os.path as osp
import json
import argparse
import pickle


class ImageDataset(Dataset):
    def __init__(
        self, img_dir, img_list,
        img_scale=(1000, 600)):
        # prefix of image path
        if isinstance(img_list, list):
            self.img_list = img_list
        else:
            self.img_list = [x.strip() for x in open(img_list)]
        self.num_img = len(self.img_list)
        self.img_dir = img_dir
        
    def __len__(self):
        return self.num_img

    def __getitem__(self, idx):
        img_id = self.img_list[idx].split('.')[0]
        img_path = osp.join(self.img_dir, self.img_list[idx])
        return img_id, img_path


def worker_func(model_cls, model_kwargs, checkpoint, dataset, data_func,
                gpu_id, idx_queue, result_queue):
    torch.cuda.set_device(gpu_id)
    model = model_cls(gpu_id=gpu_id, **model_kwargs)
    while True:
        idx = idx_queue.get()
        data = dataset[idx]
        img_id = data[0]
        bboxes, landmarks = model.detect_faces(
            data_func(data),
            thresholds=[0.5, 0.6, 0.8],
            nms_thresholds=[0.7, 0.7, 0.6])
        feats = model.easy_extract_features(
            data_func(data), landmarks)
        ret = (bboxes, landmarks, feats)
        result_queue.put((img_id, ret))


def parallel_test(model_cls,
                  model_kwargs,
                  checkpoint,
                  dataset,
                  data_func,
                  gpus,
                  worker_per_gpu=1):
    ctx = multiprocessing.get_context('spawn')
    idx_queue = ctx.Queue()
    result_queue = ctx.Queue()
    num_workers = len(gpus) * worker_per_gpu
    workers = [
        ctx.Process(
            target=worker_func,
            args=(model_cls, model_kwargs, checkpoint, dataset, data_func,
                  gpus[i % len(gpus)], idx_queue, result_queue))
        for i in range(num_workers)
    ]
    for w in workers:
        w.daemon = True
        w.start()

    for i in range(len(dataset)):
        idx_queue.put(i)

    results = {}
    prog_bar = ProgressBar(task_num=len(dataset))
    for _ in range(len(dataset)):
        img_id, res = result_queue.get()
        results[img_id] = format_ret(res)
        prog_bar.update()
    print('\n')
    for worker in workers:
        worker.terminate()

    return results


def _data_func(data):
    _, img_path = data
    img_var = img = Image.open(img_path).convert('RGB')
    return img_var


def format_ret(ret):
    bboxes, landmarks, feats = ret
    face_num = len(bboxes)
    faces = []
    for i in range(face_num):
        faces.append({
            'bbox': bboxes[i][:4],
            'conf': bboxes[i][-1],
            'landmark': landmarks[i],
            'feat': feats[i]
        })
    return faces


def test_multi_gpu(args):
    model_cls = getattr(utils, 'FaceNet')
    model_args = {'model_dir': './model'}
    dataset = ImageDataset(
        img_dir=args['img_dir'],
        img_list=args['img_list']
    )
    det_results = parallel_test(
            model_cls,
            model_args,
            None,
            dataset,
            _data_func,
            args['gpus'],
            worker_per_gpu=args['workers'])
    
    with open(args['save_path'], 'wb') as f:
        pickle.dump(det_results, f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='./data/lfw/image')
    parser.add_argument('--img_list', type=str, default='./data/lfw/meta/verify.txt')
    parser.add_argument('--save_dir', type=str, default='./data/lfw')
    parser.add_argument('--st', type=int, default=0)
    parser.add_argument('--end', type=int, default=99999999999)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,])
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    img_list = [x.strip() for x in open(args.img_list)]
    args.end = int(min(len(img_list), args.end))
    save_name = 'face_{}_{:07d}-{:07d}.pkl'.format(args.img_list.split('/')[-1].split('.')[0], args.st, args.end)
    save_path = osp.join(args.save_dir, save_name)
    img_list = img_list[args.st:args.end]
    arg_dict = {
        'img_dir': args.img_dir, 
        'img_list': img_list,
        'gpus': args.gpus,
        'workers': args.workers,
        'save_path': save_path
    }
    print(
        'img_dir: {}\n'
        'img_list_file: {}\n'
        'save_path: {}\n'
        'samples idx: {:7d} - {:7d} ({:7d})\n'
        'gpus: {}\n'
        'workers: {}\n'
        .format(
            args.img_dir, args.img_list, save_path,
            args.st, args.end, len(img_list),
            args.gpus, args.workers)
    )
    return arg_dict


def main():
    args = get_args()
    test_multi_gpu(args)

if __name__ == '__main__':
    main()