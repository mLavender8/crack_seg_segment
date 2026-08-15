import json
import os
from datetime import datetime
from pathlib import Path
import yaml
import torch

from ultralytics import YOLO


class Trainer1:
    def __init__(self,data_path,params_path,model_name='yolo11n-seg.pt'):
        self.model_name=model_name
        self.data_path=data_path
        self.params_path=params_path
        self.result=None
        self.model=None

        self.output_path=Path('runs/crack_seg1')
        self.output_path.mkdir(exist_ok=True,parents=True)

        self.train_config=self.load_params()

    def load_params(self):
        if not os.path.exists(self.params_path):
            raise FileNotFoundError(f'配置文件路径不存在:{self.params_path}')
        with open(self.params_path,'r',encoding='utf-8') as f:
            config=yaml.safe_load(f)

        config['data']=self.data_path
        config['task']='segment'
        config['model']=self.model_name
        config['device']='cuda' if torch.cuda.is_available() else 'cpu'
        config['workers']=8 if torch.cuda.is_available() else 4
        config['project']=str(self.output_path)
        config['name']=f'crack_seg_{datetime.now().strftime("%Y%m%d-%H%M%S")}'

        config.setdefault('save',True)
        config.setdefault('val',True)
        config.setdefault('verbose',True)

        return config
    def check_local_model(self):
        local_model_path=[f'models/{self.model_name}',f'./{self.model_name}',self.model_name]

        for path in local_model_path:
            if os.path.exists(path):
                print(f'本地模型已找到:{path}')
                return path
        print(f'本地模型没找到,会自动下载:{self.model_name}')
        return self.model_name
    def load_model(self):
        model_path=self.check_local_model()
        print(f'正在加载模型:{model_path}')

        if torch.cuda.is_available():
            print(f'检测到GPU：{torch.cuda.get_device_name(0)}')
            print(f'GPU内存:{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
        else:
            print(f'GPU无法使用')
        self.model=YOLO(model_path)
        print(f'模型加载完毕,设备:{self.train_config["device"]}')

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark=True
            torch.backends.cudnn.deterministic=False

            get_memory=torch.cuda.get_device_properties(0).total_memory/1024**3

            if get_memory<4:
                self.train_config['batch']=8
                print(f'GPU内存较小,调整批次为8')
            elif get_memory<8:
                self.train_config['batch']=12
                print(f'GPU内存中等,调整批次为12')
            else:
                self.train_config['batch']=16
                print(f'GPU内存较大,调整批次为16')
    def validate_dataset(self):
        print(f'验证数据集配置....')

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f'数据路径错误:{self.data_path}')

        with open(self.data_path,'r',encoding='utf-8') as f:
            data_config=yaml.safe_load(f)

        raw_base = Path(data_config["path"])
        if raw_base.is_absolute():
            base = raw_base
        elif (Path.cwd() / raw_base).exists():
            base = Path.cwd() / raw_base
        elif (Path.cwd() / "datasets" / raw_base).exists():
            base = Path.cwd() / "datasets" / raw_base
        else:
            base = raw_base
        train_path=base/'images'/'train'
        val_path=base/'images'/'val'
        test_path=base/'images'/'test'

        for name,path in [('训练集',train_path),('验证集',val_path),('测试集',test_path)]:
            if  os.path.exists(path):
                image_count=len(list(path.glob('*.jpg')))+len(list(path.glob('*.png')))
                print(f'{name}:{image_count}张图片')
            else:
                print(f'警告{name}不存在:{path}')
        label_path=base/'labels'
        train_label=label_path/'train'
        val_label=label_path/'val'
        test_label=label_path/'test'
        for name,path in [('训练集',train_label),('验证集',val_label),('测试集',test_label)]:
            if os.path.exists(path):
                label_count=len(list(path.glob('*.txt')))
                print(f'{name}:{label_count}个标签')
            else:
                print(f'警告{name}:不存在{path}')
        print(f'验证数据集配置完成')
        return True
    def train(self):
        print(f'模型训练开始')
        print(f'训练模型配置:{json.dumps(self.train_config,indent=2,ensure_ascii=False)}')

        self.result=self.model.train(**self.train_config)
        print(f'训练完成')
        return self.result
    def validate(self):
        print(f'模型验证开始')
        validate_result=self.model.val()
        print(f'验证完成')
        return validate_result
    def print_train_summary(self):
        if self.result is None:
            print(f'没有结果可以显示')
            return

        print("\n"+"="*20)
        print(f'训练结果摘要')

        try:
            results_dict=self.result.results_dict

            # 打印最终指标 - 添加安全检查
            if 'metrics/mAP50(B)' in results_dict and results_dict['metrics/mAP50(B)']:
                map50_data = results_dict['metrics/mAP50(B)']
                if isinstance(map50_data, (list, tuple)) and len(map50_data) > 0:
                    final_map50 = map50_data[-1]
                    print(f"最终检测mAP@0.5: {final_map50:.4f}")
                elif isinstance(map50_data, (int, float)):
                    print(f"最终检测mAP@0.5: {map50_data:.4f}")

            if 'metrics/mAP50-95(B)' in results_dict and results_dict['metrics/mAP50-95(B)']:
                map50_95_data = results_dict['metrics/mAP50-95(B)']
                if isinstance(map50_95_data, (list, tuple)) and len(map50_95_data) > 0:
                    final_map50_95 = map50_95_data[-1]
                    print(f"最终检测mAP@0.5:0.95: {final_map50_95:.4f}")
                elif isinstance(map50_95_data, (int, float)):
                    print(f"最终检测mAP@0.5:0.95: {map50_95_data:.4f}")

            if 'metrics/mAP50(M)' in results_dict and results_dict['metrics/mAP50(M)']:
                seg_map50_data = results_dict['metrics/mAP50(M)']
                if isinstance(seg_map50_data, (list, tuple)) and len(seg_map50_data) > 0:
                    final_seg_map50 = seg_map50_data[-1]
                    print(f"最终分割mAP@0.5: {final_seg_map50:.4f}")
                elif isinstance(seg_map50_data, (int, float)):
                    print(f"最终分割mAP@0.5: {seg_map50_data:.4f}")

            if 'metrics/mAP50-95(M)' in results_dict and results_dict['metrics/mAP50-95(M)']:
                seg_map50_95_data = results_dict['metrics/mAP50-95(M)']
                if isinstance(seg_map50_95_data, (list, tuple)) and len(seg_map50_95_data) > 0:
                    final_seg_map50_95 = seg_map50_95_data[-1]
                    print(f"最终分割mAP@0.5:0.95: {final_seg_map50_95:.4f}")
                elif isinstance(seg_map50_95_data, (int, float)):
                    print(f"最终分割mAP@0.5:0.95: {seg_map50_95_data:.4f}")

            # 打印最终损失
            if 'train/box_loss' in results_dict and results_dict['train/box_loss']:
                box_loss_data = results_dict['train/box_loss']
                if isinstance(box_loss_data, (list, tuple)) and len(box_loss_data) > 0:
                    final_box_loss = box_loss_data[-1]
                    print(f"最终边界框损失: {final_box_loss:.4f}")
                elif isinstance(box_loss_data, (int, float)):
                    print(f"最终边界框损失: {box_loss_data:.4f}")

            if 'train/seg_loss' in results_dict and results_dict['train/seg_loss']:
                seg_loss_data = results_dict['train/seg_loss']
                if isinstance(seg_loss_data, (list, tuple)) and len(seg_loss_data) > 0:
                    final_seg_loss = seg_loss_data[-1]
                    print(f"最终分割损失: {final_seg_loss:.4f}")
                elif isinstance(seg_loss_data, (int, float)):
                    print(f"最终分割损失: {seg_loss_data:.4f}")

            print("=" * 50)

        except Exception as e:
            print(f"显示训练结果时出错: {e}")
            print("训练已完成，模型已保存，可以查看训练日志获取详细信息")

    def save_training_info(self):
        info={
            'model_name':self.model_name,
            'data_path':self.data_path,
            'params_path':self.params_path,
            'train_config':self.train_config,
            'device':self.train_config['device']
        }
        info_path=self.output_path/'train_info1.json'
        with open(info_path, 'w',encoding='utf-8') as f:
            json.dump(info,f,indent=2,ensure_ascii=False)

        print(f'信息已经保存到：{info_path}')

def main():
    print("=" * 60)
    print("YOLO11-seg 裂隙分割训练程序")
    print("=" * 60)

    data_path1=r'data_path.yaml'
    params_path1=r'params_path.yaml'

    if not os.path.exists(data_path1):
        print(f"错误: 数据集配置文件不存在: {data_path1}")
        print("请确保数据集已正确解压到当前目录")
        return

    try:
        trainer=Trainer1(params_path=params_path1,data_path=data_path1)
        trainer.validate_dataset()
        trainer.load_model()
        trainer.train()
        trainer.validate()
        trainer.print_train_summary()
        trainer.save_training_info()

        print("\n" + "=" * 60)
        print("训练完成! 模型已保存到 runs/crack-seg 目录")
        print("=" * 60)

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
if __name__ == '__main__':
    main()