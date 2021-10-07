
import os
import cv2
from tqdm import tqdm

import insightface
from insightface.app import FaceAnalysis


app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 480))


head_labels = {1: "unsure",
               2: "front",
               3: "right",
               4: "left"}
def make_labels(annot_path, image_path, make_root):
    with open(annot_path) as f:
        data = f.readlines()
        for d in tqdm(data):
            d_dict = eval(d)
            make_path = os.path.join(make_root, d_dict["ID"] + ".txt")
            img_path = os.path.join(image_path, d_dict["ID"] + ".jpg")
            tmp_img = cv2.imread(img_path)
            copy_img = tmp_img.copy()
            h, w, _ = tmp_img.shape

            txt = ""
            gtboxes = d_dict["gtboxes"]
            n = 0
            for gtbox in gtboxes:
                if gtbox["tag"] != "person":
                    continue
                full_body = gtbox["fbox"]
                visible_body = gtbox["vbox"]
                fin_xyxy = get_fin_xyxy(full_body, visible_body, (w, h))

                head = gtbox["hbox"]
                head_xyxy = get_fin_xyxy(head, None, (w, h))

                #body_occ = gtbox["extra"]["occ"] if "occ" in gtbox["extra"] else 0

                body_txt, head_txt = "", ""
                fin_box, head_box = "", ""
                if fin_xyxy:
                    fin_box = xyxy2cpwh(fin_xyxy, (w, h))
                    if fin_box:
                        body_txt = f"0 {fin_box[0]} {fin_box[1]} {fin_box[2]} {fin_box[3]}\n"
                        '''cv2.rectangle(tmp_img, (fin_xyxy[0], fin_xyxy[1]), (fin_xyxy[2], fin_xyxy[3]),
                                      color=(0, 225, 225),
                                      thickness=2)
                        label = "body_occ" if body_occ else "body"
                        t_size = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=1)[0]
                        cv2.rectangle(tmp_img, (fin_xyxy[0], fin_xyxy[1]),
                                      (fin_xyxy[0] + t_size[0], fin_xyxy[1] - t_size[1] - 3), (0, 255, 255), -1,
                                      cv2.LINE_AA)
                        cv2.putText(tmp_img, label, (fin_xyxy[0], fin_xyxy[1] - 2), 0, 2 / 3, [0, 0, 0], thickness=1,
                                    lineType=cv2.LINE_AA)'''
                if head_xyxy:
                    head_box = xyxy2cpwh(head_xyxy, (w, h))
                    if head_box:
                        head_img = copy_img[head_xyxy[1]: head_xyxy[3], head_xyxy[0]: head_xyxy[2]]
                        face = app.get(head_img)

                        if len(face) >= 1:
                            # head_idx = get_head_idx(face[0], head_img)
                            head_idx = 2
                        else:
                            head_idx = 1
                        head_label = f"head_{head_labels[head_idx]}"
                        head_txt = f"{head_idx} {head_box[0]} {head_box[1]} {head_box[2]} {head_box[3]}\n"
                        '''cv2.imshow(f"{head_label} {n}", head_img)
                        cv2.rectangle(tmp_img, (head_xyxy[0], head_xyxy[1]), (head_xyxy[2], head_xyxy[3]),
                                      color=(225, 0, 0),
                                      thickness=2)'''
                if fin_box == head_box:
                    txt += head_txt
                else:
                    txt += body_txt
                    txt += head_txt
                n += 1

            '''cv2.imshow("img", tmp_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''
            with open(make_path, "w") as s:
                s.write(txt)


# kps[0]: right eye
# kps[1]: left eye
# kps[2]: nose
# kps[3]: right mouse
# kps[4]: left mouse

# head_idx = 1: normal head (or back head)
# head_idx = 2: front head
# head_idx = 3: right head
# head_idx = 4: left head
def get_head_idx(face, head_img, mid_thr=0.25):
    kps = face["kps"]
    re = kps[0]
    le = kps[1]
    no = kps[2]
    rm = kps[3]
    lm = kps[4]
    me = [(re[0] + le[0]) / 2, (re[1] + le[1]) / 2]
    head_size = head_img.shape[:2]
    if re[0] < le[0]:
        if no[0] < head_size[1] * mid_thr:
            if le[0] < head_size[1] * (mid_thr + 0.05):
                head_idx = 4
            else:
                head_idx = 2
        elif head_size[1] * mid_thr < no[0] < head_size[1] * (1 - mid_thr):
            head_idx = 2
        else:
            if re[0] > head_size[1] * (1 - mid_thr - 0.05):
                head_idx = 3
            else:
                head_idx = 2
    else:
        head_idx = 1
    return head_idx



def xyxy2cpwh(xyxy, img_size):
    cpwh = [round((xyxy[0] + xyxy[2]) / 2 / img_size[0], 6),
            round((xyxy[1] + xyxy[3]) / 2 / img_size[1], 6),
            round((xyxy[2] - xyxy[0]) / img_size[0], 6),
            round((xyxy[3] - xyxy[1]) / img_size[1], 6)]

    for n in cpwh:
        if n > 1:
            cpwh = []
        elif n <= 0:
            cpwh = []
    return cpwh


def get_fin_xyxy(full_body, visible_body, img_size):
    if visible_body is not None:
        valid_xyxy = [min(max(min(full_body[0], visible_body[0]), 0), img_size[0]),
                      min(max(min(full_body[1], visible_body[1]), 0), img_size[1]),
                      max(min(max(full_body[0] + full_body[2], visible_body[0] + visible_body[2]), img_size[0]), 0),
                      max(min(max(full_body[1] + full_body[3], visible_body[1] + visible_body[3]), img_size[1]), 0)]
    else:
        valid_xyxy = [min(max(full_body[0], 0), img_size[0]),
                      min(max(full_body[1], 0), img_size[1]),
                      max(min(full_body[0] + full_body[2], img_size[0]), 0),
                      max(min(full_body[1] + full_body[3], img_size[1]), 0)]

    for n in valid_xyxy[2:]:
        if n <= 0:
            valid_xyxy = []
            break
    return valid_xyxy


if __name__ == "__main__":
    root = "/media/daton/D6A88B27A88B0569/dataset/crowdhuman"

    train_image_path = os.path.join(root, "images", "train")
    valid_image_path = os.path.join(root, "images", "valid")

    train_annot_path = os.path.join(root, "annotation_train.odgt")
    train_make_root = os.path.join(root, "labels_front_unsure", "train")
    if not os.path.isdir(train_make_root):
        os.mkdir(train_make_root)

    valid_annot_path = os.path.join(root, "annotation_val.odgt")
    valid_make_root = os.path.join(root, "labels_front_unsure", "valid")
    if not os.path.isdir(valid_make_root):
        os.mkdir(valid_make_root)

    make_labels(train_annot_path, train_image_path, train_make_root)
    make_labels(valid_annot_path, valid_image_path, valid_make_root)