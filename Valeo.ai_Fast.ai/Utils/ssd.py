import keras
import numpy as np

from keras import backend as K

class SSD_LOSS:
    '''	Loss Class for SSD model.

    # Arguments
        k: The number of anchor per activation
        nbClass: The number of class
        anchors: coordinates of the anchors [x_center, y_center, width, height] 
        grid_sizes: size of the grid
    '''

    def __init__(self,k,nbClass,anchors,grid_sizes):  
        self.k = k
        self.nb_activation = len(grid_sizes)
        self.nb_class = nbClass
        self.anchors = anchors
        self.anchor_cnr = K.tf.to_float(self.hw2corners(anchors[:,:2], anchors[:,2:]))
        self.grid_sizes = K.tf.to_float(grid_sizes)
    
    def hw2corners(self,ctr, hw): return K.concatenate((ctr-hw/2, ctr+hw/2), axis=1)

    def actn_to_bb(self,actn):
        actn_bbs = K.tf.tanh(actn)
        actn_centers = (actn_bbs[:,:2]/2 * self.grid_sizes) + self.anchors[:,:2]
        actn_hw = (actn_bbs[:,2:]/2+1) * self.anchors[:,2:]
        return self.hw2corners(actn_centers, actn_hw)

    def box_sz(self,b): return ((b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1]))

    def intersect(self,box_a, box_b):

        max_xy = K.minimum(box_a[:,None,2:], box_b[:, 2:])
        min_xy = K.maximum(box_a[:,None,:2], box_b[:, :2])

        inter = K.clip((max_xy - min_xy),0,None)
        return inter[:, :, 0] * inter[:, :, 1]

    def jaccard(self,box_a, box_b):
        box_a = K.cast(box_a,K.tf.float64)
        box_b = K.cast(box_b,K.tf.float64)
        inter = self.intersect(box_a, box_b) 
        inter = K.cast(inter,K.tf.float64)
        union = K.cast(K.expand_dims(self.box_sz(box_a),1),K.tf.float64)+ K.cast(K.expand_dims(self.box_sz(box_b),0),K.tf.float64) - inter  

        return inter/union

    def get_y(self,y):
        #y = [class y0 x0 y1 x1]
        diff = y[:,3]-y[:,1]+y[:,4]-y[:,2]
        idx = K.tf.where(diff>0)
        return K.tf.gather_nd(y,idx)

    def SSD_1_LOSS(self,yGT, yPred):
        
        # convert to matrix
        yGT = K.reshape(yGT,[20,5])
        
        # GT has a fixed size, so we keep only non zero box
        yGT = self.get_y(yGT)
        
        # split BBox and Class
        bbox_GT = yGT[:,1:5]
        class_GT = yGT[:,0]
        
        # split class and box
        
        yPredClas = yPred[:self.nb_activation*self.nb_class]
        yPredBox = yPred[self.nb_activation*self.nb_class:]
        class_Pred = K.reshape(yPredClas,[self.nb_activation,self.nb_class])
        bbox_Pred = K.reshape(yPredBox,[self.nb_activation,4])
        
        # convert to activation
        a_ic = self.actn_to_bb(bbox_Pred)
          
        # ccompute jaccard matrix
        overlaps = self.jaccard(bbox_GT, self.anchor_cnr)

        # Map to ground truth
        gt_idx = K.argmax(overlaps,0) # [16] for each activation, ID of the GT with the best overlapp   
        gt_overlap = K.max(overlaps,0) # [16] for each cell, ID of the GT with the best overlapp 
        prior_idx = K.argmax(overlaps,1) # [4] for each GT, ID of best anchors    
        prior_overlap = K.max(overlaps,1) # [4] for each GT, value of tye best overlapp

        # BBOX Loss
        ADD = K.tf.one_hot(prior_idx,self.nb_activation)
        ADD = K.cast(K.sum(ADD,axis=0),('float64'))

        gt_overlap = gt_overlap+ADD
        
        Threshold = 0.4
        valid_anchor = gt_overlap>Threshold   
        mask = K.cast(valid_anchor,('float32'))   
        bbox = K.gather(bbox_GT,gt_idx)    
        bbox = K.cast(bbox,('float32'))

        loc_loss = K.abs(a_ic-bbox)     
        loc_loss = K.sum(loc_loss,axis=1)   
        loc_loss = K.tf.multiply(loc_loss,mask)
        loc_loss = (K.sum(loc_loss))/K.sum(mask)     
               
	# Classification Loss
  
        # Loss for overlapp >0.5
        sel_gt_clas = K.gather(class_GT,gt_idx)
        gt_class_per_activation = K.one_hot(K.cast(sel_gt_clas,('int32')), 20)
        
        # for overlapp below 0.5, we have to put 0 in gt_class_per_activation
        valid_anchor = gt_overlap>Threshold 
        mask = K.cast(valid_anchor,('float32'))
        mask = K.reshape(K.repeat_elements(mask,self.nb_class,0),(self.nb_activation,self.nb_class))

        One_Hot_Overlap = gt_class_per_activation*mask
                
        # then, we estimate BCE for mandatory box (GT)
        pred_mandatory_anchor = K.gather(class_Pred,prior_idx)
        One_Hot_mandatory = K.tf.one_hot(K.cast(class_GT,'int32'),self.nb_class)
        
        target=K.concatenate([One_Hot_Overlap,One_Hot_mandatory],axis=0)
        pred=K.concatenate([class_Pred,pred_mandatory_anchor],axis=0)

        clas_loss = K.mean(K.binary_crossentropy(target, pred))
        
        return clas_loss*5 + loc_loss
    
    def compute_loss(self,y_true,y_pred):

        # convert to tensor:
        y_true_ts = K.tf.to_float(y_true) # [batch, nb_max_gt 20 , 1+4+1]
        y_pred_ts = K.tf.to_float(y_pred) # [batch, nb_activation 16 , (20 + 4)]   

        elements = (y_true_ts, y_pred_ts)
        loss = K.tf.map_fn(
                    lambda x:self.SSD_1_LOSS(x[0],x[1]), elements, dtype=K.tf.float32)


        return K.sum(loss)

class SSD_FOCAL_LOSS:
    '''	Loss Class for SSD model.

    # Arguments
        k: The number of anchor per activation
        nbClass: The number of class
        anchors: coordinates of the anchors [x_center, y_center, width, height] 
        grid_sizes: size of the grid
    '''

    def __init__(self,k,nbClass,anchors,grid_sizes):  
        self.k = k
        self.nb_activation = len(grid_sizes)
        self.nb_class = nbClass
        self.anchors = anchors
        self.anchor_cnr = K.tf.to_float(self.hw2corners(anchors[:,:2], anchors[:,2:]))        
        self.grid_sizes = K.tf.to_float(grid_sizes)
    
    def hw2corners(self,ctr, hw): return K.concatenate((ctr-hw/2, ctr+hw/2), axis=1)

    def actn_to_bb(self,actn):
        actn_bbs = K.tf.tanh(actn)
        actn_centers = (actn_bbs[:,:2]/2 * self.grid_sizes) + self.anchors[:,:2]
        actn_hw = (actn_bbs[:,2:]/2+1) * self.anchors[:,2:]
        return self.hw2corners(actn_centers, actn_hw)

    def box_sz(self,b): return ((b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1]))

    def intersect(self,box_a, box_b):

        max_xy = K.minimum(box_a[:,None,2:], box_b[:, 2:])
        min_xy = K.maximum(box_a[:,None,:2], box_b[:, :2])

        inter = K.clip((max_xy - min_xy),0,None)
        return inter[:, :, 0] * inter[:, :, 1]

    def jaccard(self,box_a, box_b):
        box_a = K.cast(box_a,K.tf.float64)
        box_b = K.cast(box_b,K.tf.float64)
        inter = self.intersect(box_a, box_b) 
        inter = K.cast(inter,K.tf.float64)
        union = K.cast(K.expand_dims(self.box_sz(box_a),1),K.tf.float64)+ K.cast(K.expand_dims(self.box_sz(box_b),0),K.tf.float64) - inter  

        return inter/union

    def get_y(self,y):
        #y = [class y0 x0 y1 x1]
        diff = y[:,3]-y[:,1]+y[:,4]-y[:,2]
        idx = K.tf.where(diff>0)
        return K.tf.gather_nd(y,idx)
    
    def get_weight(self,pred,target):
        alpha,gamma = 0.9,4

        # Convert to proba
        proba = pred*target + (1-pred)*(1-target)
        
        # Weight for unbalance example (much more background)
        weight_unbalanced = (alpha*target + (1-alpha)*(1-target))*40
        
        # weight for hard example (focal loss)
        weight_hard = K.pow((1-proba),gamma) * 40
        
        return weight_hard*weight_unbalanced

    def SSD_1_LOSS(self,yGT, yPred):
        
        # convert to matrix
        yGT = K.reshape(yGT,[20,5])
        
        # GT has a fixed size, so we keep only non zero box
        yGT = self.get_y(yGT)
        
        # split BBox and Class
        bbox_GT = yGT[:,1:5]
        class_GT = yGT[:,0]
        
        # split class and box
        
        yPredClas = yPred[:self.nb_activation*self.nb_class]
        yPredBox = yPred[self.nb_activation*self.nb_class:]
        class_Pred = K.reshape(yPredClas,[self.nb_activation,self.nb_class])
        bbox_Pred = K.reshape(yPredBox,[self.nb_activation,4])
        
        # convert to activation
        a_ic = self.actn_to_bb(bbox_Pred)
          
        # ccompute jaccard matrix
        overlaps = self.jaccard(bbox_GT, self.anchor_cnr)

        # Map to ground truth
        gt_idx = K.argmax(overlaps,0) # [16] for each activation, ID of the GT with the best overlapp   
        gt_overlap = K.max(overlaps,0) # [16] for each cell, ID of the GT with the best overlapp 
        prior_idx = K.argmax(overlaps,1) # [4] for each GT, ID of best anchors    
        prior_overlap = K.max(overlaps,1) # [4] for each GT, value of tye best overlapp

        # BBOX Loss
        ADD = K.tf.one_hot(prior_idx,self.nb_activation)
        ADD = K.cast(K.sum(ADD,axis=0),('float64'))

        gt_overlap = gt_overlap+ADD
        
        Threshold = 0.4
        valid_anchor = gt_overlap>Threshold   
        mask = K.cast(valid_anchor,('float32'))   
        bbox = K.gather(bbox_GT,gt_idx)    
        bbox = K.cast(bbox,('float32'))

        loc_loss = K.abs(a_ic-bbox)     
        loc_loss = K.sum(loc_loss,axis=1)   
        loc_loss = K.tf.multiply(loc_loss,mask)
        loc_loss = (K.sum(loc_loss))/K.sum(mask)     
               
	# Classification Loss
  
        # Loss for overlapp >0.5
        sel_gt_clas = K.gather(class_GT,gt_idx)
        gt_class_per_activation = K.one_hot(K.cast(sel_gt_clas,('int32')), 20)
        
        # for overlapp below 0.5, we have to put 0 in gt_class_per_activation
        valid_anchor = gt_overlap>Threshold 
        mask = K.cast(valid_anchor,('float32'))
        mask = K.reshape(K.repeat_elements(mask,self.nb_class,0),(self.nb_activation,self.nb_class))

        One_Hot_Overlap = gt_class_per_activation*mask
                
        # then, we estimate BCE for mandatory box (GT)
        pred_mandatory_anchor = K.gather(class_Pred,prior_idx)
        One_Hot_mandatory = K.tf.one_hot(K.cast(class_GT,'int32'),self.nb_class)
        
        target=K.concatenate([One_Hot_Overlap,One_Hot_mandatory],axis=0)
        pred=K.concatenate([class_Pred,pred_mandatory_anchor],axis=0)

        weight = self.get_weight(pred,target)

        clas_loss = K.mean(K.binary_crossentropy(target, pred,from_logits=True)*weight)
        
        return clas_loss/2 + loc_loss
    
    def compute_loss(self,y_true,y_pred):

        # convert to tensor:
        y_true_ts = K.tf.to_float(y_true) # [batch, nb_max_gt 20 , 1+4+1]
        y_pred_ts = K.tf.to_float(y_pred) # [batch, nb_activation 16 , (20 + 4)]   

        elements = (y_true_ts, y_pred_ts)
        loss = K.tf.map_fn(
                    lambda x:self.SSD_1_LOSS(x[0],x[1]), elements, dtype=K.tf.float32)


        return K.sum(loss)

def nms(boxes, scores, overlap=0.5, top_k=100):
    keep = boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1)*(y2 - y1)
    idx = np.argsort(scores)
    v = scores[idx]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1=yy1=xx2=yy2=w=h=boxes

    keep=[]
    while len(idx) > 0:
        i = idx[-1]  # index of current largest val
        keep.append(i)
        if len(idx) == 1: break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        
        xx1 = x1[idx]
        yy1 = y1[idx]
        xx2 = x2[idx]
        yy2 = y2[idx]
        
        # store element-wise max with next highest score
        xx1 = np.clip(xx1, x1[i],None)
        yy1 = np.clip(yy1, y1[i],None)
        xx2 = np.clip(xx2, None,x2[i])
        yy2 = np.clip(yy2, None,y2[i])
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = np.clip(w, 0.0,None)
        h = np.clip(h, 0.0,None)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = area[idx]  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap

        idx = idx[np.where(IoU<=overlap)]
    return keep
