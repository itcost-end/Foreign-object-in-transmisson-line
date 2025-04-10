import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
      #model = YOLO('/opt/data/private/HanCi/yolov8-main/ultralytics-main/runs/95.2(第2次跑)/train-origin-MADRMELA(12)-RFSAConv(backbone)-LSCSBD/weights/best.pt') # select your model.pt path
      model = YOLO('/opt/data/private/HanCi/visdeone/runs/4types dataset/train-origin-C2f-RFCAConv+LSCD2/weights/best.pt')
    # model.predict(source='/opt/data/private/HanCi/yolov8-main/ultralytics-main/dataset(721)/images/train-origin/kuochong_qiqiu142.jpg',
    #  model.predict(source='/opt/data/private/HanCi/yolov8-main/ultralytics-main/dataset(721)/images/test-origin/kuochong_FZ205.jpg',
    # model.predict(source='/opt/data/private/HanCi/yolov8-main/ultralytics-main/dataset(721)/images/train-origin/plastic (3).jpg',
    # model.predict(source='/opt/data/private/HanCi/yolov8-main/ultralytics-main/dataset(721)/images/train-origin/banner (88).jpg',
      model.predict(source='/opt/data/private/HanCi/visdeone/4 types dataset/images/train-origin/000005.jpg',

                  imgsz=640,
                  project='runs/detect',
                  name='Ours algorithm',
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )