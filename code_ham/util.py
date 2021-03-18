from skimage import feature
import medpy.metric as mm
import numpy as np

def get_center(mask):
    cand=np.where(mask==1)
    if len(cand[1])==0:
        return [0,0]
    x_min=min(cand[1])
    x_max=max(cand[1])
    y_min=min(cand[0])
    y_max=max(cand[0])
    center = [(x_min+x_max)/2, (y_min+y_max)/2]
    return center

def getcircum(mask):
    edg = feature.canny(mask,low_threshold=1,high_threshold=1)
    cand = np.where(edg==True)
    return len(cand[0])

def metrics(mas,pred):
    cir, dis = [], []
    iou, dice = [], []
    avd, vs = [], []
    for i in range(len(pred)):
        mas_ = mas[i]
        pred_ = pred[i]
        
        result = np.atleast_1d(pred_.astype(np.bool))
        reference = np.atleast_1d(mas_.astype(np.bool))
        tp = np.count_nonzero(result & reference)
        fn = np.count_nonzero(~result & reference)
        fp = np.count_nonzero(result & ~reference)
        vs.append(1 - abs(fn-fp)/(2*tp+fp+fn))

        true = getcircum(mas_)
        pre = getcircum(pred_)
        cir.append(abs(true-pre))

        inter = np.logical_and(pred_==1,mas_==1)
        union  = np.logical_or(pred_==1,mas_==1)
        iou.append(np.sum(inter)/np.sum(union))
        dice.append(np.sum(pred_[mas_==1])*2.0 / (np.sum(pred_) + np.sum(mas_)))
        if (pred_==0).all():
            avd.append(float('-inf'))
            dis.append(float('-inf'))
        else:
            avd.append(mm.binary.hd(mas_,pred_))
            true = np.array(get_center(mas_))
            pre = np.array(get_center(pred_))
            dis.append(np.linalg.norm(true-pre))

    dm=max(dis)
    am=max(avd)
    dis = [dm if item == float('-inf') else item for item in dis]
    avd = [am if item == float('-inf') else item for item in avd]
    print("Best test IOU is", np.mean(iou))
    print("Best test DICE is", np.mean(dice))
    print('Test center error is', np.mean(dis))
    print('Test circumstance error is', np.mean(cir)/2)
    print('Test AVD is ', np.mean(avd))
    print('Test VS is ', np.mean(vs))
