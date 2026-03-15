import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ISICDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(csv_path)
        self.transform = transform

        # 라벨의 8개 진단명 저장
        self.label_columns = [
            'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

        # 이미지 파일명 리스트 저장
        self.image_names = self.labels_df['image'].values

        # 이미지의 정답을 one-hot 벡터로
        self.labels = self.labels_df[self.label_columns].values

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx] + ".jpg"
        img_path = os.path.join(self.image_dir, img_name)

        # 이미지 로딩 (RGB로 통일)
        image = Image.open(img_path).convert('RGB')

        # 전처리 (크기 조정, 정규화)
        if self.transform:
            image = self.transform(image)

        # 라벨-> numpy array로
        label = self.labels[idx].astype('float32')

        return image, label


if __name__ == "__main__":
    import torchvision.transforms as transforms

    # 전처리 정의
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 데이터 경로
    image_dir = "./data/ISIC_2019_Training_Input"
    csv_path = "./data/ISIC_2019_Training_GroundTruth.csv"

    # 데이터셋 인스턴스 생성
    dataset = ISICDataset(image_dir, csv_path, transform=transform)

    # 샘플 1개 확인
    print("전체 이미지 수:", len(dataset))
    img, label = dataset[0]
    print("이미지 크기:", img.shape)
    print("라벨:", label)