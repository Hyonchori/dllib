
import os
from tqdm import tqdm


def make_labels(annot_path, image_path, make_root):
    with open(annot_path) as f:
        data = f.readlines()
        for d in tqdm(data):
            d_dict = eval(d)
            make_path = os.path.join(make_root, d_dict["ID"] + ".txt")
            img_path = os.path.join(image_path, d_dict["ID"] + ".jpg")
            tmp_img = cv2.imread(img_path)
            h, w, _ = tmp_img.shape

            txt = ""
            gtboxes = d_dict["gtboxes"]
            for gtbox in gtboxes:
                if gtbox["tag"] != "person":
                    continue
                print(gtbox)
                full_body = gtbox["fbox"]
                visible_body = gtbox["vbox"]
                fin_xyxy = get_fin_xyxy(full_body, visible_body, (w, h))

                head = gtbox["hbox"]
                head_xyxy = get_fin_xyxy(head, None, (w, h))

                body_occ = gtbox["extra"]["occ"] if "occ" in gtbox["extra"] else 0

                if fin_xyxy:
                    fin_box = xyxy2cpwh(fin_xyxy, (w, h))
                    if fin_box:
                        txt += f"0 {fin_box[0]} {fin_box[1]} {fin_box[2]} {fin_box[3]}\n"
                        cv2.rectangle(tmp_img, (fin_xyxy[0], fin_xyxy[1]), (fin_xyxy[2], fin_xyxy[3]), color=(0, 225, 225),
                                      thickness=2)
                        label = "body_occ" if body_occ else "body"
                        t_size = cv2.getTextSize(label, 0, fontScale=2/3, thickness=1)[0]
                        cv2.rectangle(tmp_img, (fin_xyxy[0], fin_xyxy[1]), (fin_xyxy[0] + t_size[0], fin_xyxy[1] - t_size[1] - 3), (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.putText(tmp_img, label, (fin_xyxy[0], fin_xyxy[1] - 2), 0, 2/3, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)
                if head_xyxy:
                    head_box = xyxy2cpwh(head_xyxy, (w, h))
                    if head_box:
                        txt += f"1 {head_box[0]} {head_box[1]} {head_box[2]} {head_box[3]}\n"
                        cv2.rectangle(tmp_img, (head_xyxy[0], head_xyxy[1]), (head_xyxy[2], head_xyxy[3]),
                                      color=(225, 0, 0),
                                      thickness=2)

            '''with open(make_path, "w") as s:
                s.write(txt)'''

            cv2.imshow("img", tmp_img)
            cv2.waitKey(0)



def xywh2cpwh(xywh, img_size):
    cpwh = [round((xywh[0] + xywh[2] / 2) / img_size[0], 6),
            round((xywh[1] + xywh[3] / 2) / img_size[1], 6),
            round(xywh[2] / img_size[0], 6),
            round(xywh[3] / img_size[1], 6)]
    return cpwh


def xyxy2cpwh(xyxy, img_size):
    cpwh = [round((xyxy[0] + xyxy[2]) / 2 / img_size[0], 6),
            round((xyxy[1] + xyxy[3]) / 2 / img_size[1], 6),
            round((xyxy[2] - xyxy[0]) / img_size[0], 6),
            round((xyxy[3] - xyxy[1]) / img_size[1], 6)]

    for n in cpwh:
        if n > 1:
            '''print(f"\nimg size: {img_size}")
            print(f"out of range: {xyxy} -> {cpwh}")'''
            cpwh = []
        elif n <= 0:
            '''print(f"\nimg size: {img_size}")
            print(f"negative: {xyxy} -> {cpwh}")'''
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
    import cv2

    root = "/media/daton/D6A88B27A88B0569/dataset/crowdhuman"

    train_image_path = os.path.join(root, "images", "train")
    valid_image_path = os.path.join(root, "images", "valid")

    train_annot_path = os.path.join(root, "annotation_train.odgt")
    train_make_root = os.path.join(root, "labels", "train")
    print(len(os.listdir(train_make_root)))

    valid_annot_path = os.path.join(root, "annotation_val.odgt")
    valid_make_root = os.path.join(root, "labels", "valid")

    make_labels(train_annot_path, train_image_path, train_make_root)
    make_labels(valid_annot_path, valid_image_path, valid_make_root)


