from common_imports import *


class EffDetDataset(Dataset):
    def __init__(self, data, img_size=512):
        self.data = data
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        with open(item['image'], 'rb') as f:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        img = cv2.resize(img, (self.img_size, self.img_size))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        bbox = item['bbox']
        bbox_scaled = [
            bbox[0] * self.img_size / w,
            bbox[1] * self.img_size / h,
            bbox[2] * self.img_size / w,
            bbox[3] * self.img_size / h
        ]

        return img_tensor, {
            'bbox': torch.tensor([bbox_scaled], dtype=torch.float32),
            'cls': torch.tensor([item['label']], dtype=torch.long),
            'img_scale': torch.tensor([1.0], dtype=torch.float32),
            'img_size': torch.tensor([self.img_size, self.img_size], dtype=torch.long)
        }


def collate_fn(batch):
    images = torch.stack([x[0] for x in batch])
    max_boxes = max([x[1]['bbox'].shape[0] for x in batch])

    bboxes, classes, scales, sizes = [], [], [], []
    for x in batch:
        bbox, cls = x[1]['bbox'], x[1]['cls']
        n = bbox.shape[0]
        if n < max_boxes:
            bbox = torch.cat([bbox, torch.zeros((max_boxes - n, 4))])
            cls = torch.cat([cls, torch.ones(max_boxes - n, dtype=torch.long) * -1])
        bboxes.append(bbox)
        classes.append(cls)
        scales.append(x[1]['img_scale'])
        sizes.append(x[1]['img_size'])

    return images, {
        'bbox': torch.stack(bboxes),
        'cls': torch.stack(classes),
        'img_scale': torch.stack(scales),
        'img_size': torch.stack(sizes)
    }