import os
import cv2
import torch
from torch.utils.data import Dataset

class RetinopathyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        """
        Args:
            df (pd.DataFrame): CSV dosyasından gelen dataframe
            img_dir (str): Resimlerin bulunduğu klasör yolu
            transform (albumentations.Compose): Albumentations augmentasyonları
        """
        self.df = df
        self.img_dir = img_dir          # ✅ artık self.img_dir mevcut
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        # ✅ FutureWarning çözümü: sütun adını kullanıyoruz
        img_id = str(row["id_code"])
        label = torch.tensor(row["diagnosis"], dtype=torch.long)

        # Resim yolu
        img_path = os.path.join(self.img_dir, f"{img_id}.png")

        # Resmi oku ve RGB'ye çevir
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Augmentation uygula
        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label
